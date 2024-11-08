#include "util.cuh"

namespace {
template <typename layoutTile, typename layoutBlock, typename layoutThread>
__global__ void gemmKernel(const float *__restrict__ A,
                           const float *__restrict__ B, float *__restrict__ C,
                           float alpha, float beta, unsigned M, unsigned N,
                           unsigned K) {
  constexpr unsigned ratio = sizeof(aduc::float4) / sizeof(float);
  using layoutTileT =
      aduc::layout<layoutTile::M / ratio, layoutTile::N / ratio,
                        layoutTile::K / ratio>;
  using layoutThreadT =
      aduc::layout<layoutThread::M / ratio, layoutThread::N / ratio>;
  constexpr unsigned blockSize = layoutBlock::M * layoutBlock::N;
  constexpr aduc::float4 float4Zero{0.f, 0.f, 0.f, 0.f};

  __shared__ aduc::float4 tileA[2][layoutTile::K][layoutTileT::M];
  __shared__ aduc::float4 tileB[2][layoutTile::K][layoutTileT::N];

  const unsigned nInTileC = threadIdx.x % layoutBlock::M;
  const unsigned mInTileC = threadIdx.x / layoutBlock::M;

  const unsigned kInTileA = threadIdx.x % layoutTileT::K;
  const unsigned mInTileA = threadIdx.x / layoutTileT::K;

  const unsigned nInTileB = threadIdx.x % layoutTileT::N;
  const unsigned kinTileB = threadIdx.x / layoutTileT::N;

  aduc::tensor2d<const aduc::float4> pA{A, M, K / ratio};
  pA.addOffset(layoutTile::M * blockIdx.y + mInTileA, kInTileA);
  aduc::tensor2d<const aduc::float4> pB{B, K, N / ratio};
  pB.addOffset(kinTileB,
               layoutTileT::N * blockIdx.x + nInTileB * layoutThreadT::N);
  aduc::tensor2d<aduc::float4> pC{C, M, N / ratio};
  pC.addOffset(layoutTile::M * blockIdx.y + mInTileC * layoutThread::M,
               layoutTileT::N * blockIdx.x + nInTileC * layoutThreadT::N);

  constexpr unsigned tileSizeA = layoutTile::M * layoutTile::K;
  constexpr unsigned tileSizeB = layoutTile::N * layoutTile::K;
  constexpr unsigned tileIterationsA = tileSizeA / blockSize / ratio;
  constexpr unsigned tileGlobalIntervalA = blockSize / layoutTileT::K;
  constexpr unsigned tileComputeIterationsA = layoutTileT::M / layoutBlock::M;
  constexpr unsigned tileSharedIntervalAT =
      layoutTileT::M / tileComputeIterationsA;
  constexpr unsigned tileIterationsB = tileSizeB / blockSize / ratio;
  constexpr unsigned tileGlobalIntervalB = blockSize / layoutTileT::N;
  constexpr unsigned tileComputeIterationsB = layoutTileT::N / layoutBlock::N;
  constexpr unsigned tileSharedIntervalBT =
      layoutTileT::N / tileComputeIterationsB;

  aduc::float4 bufferA[tileIterationsA];
  aduc::float4 bufferB[tileIterationsB];
  bool validLoadTileA[tileIterationsA];
  bool validLoadTileB[tileIterationsB];

#pragma unroll
  for (unsigned i = 0; i < tileIterationsA; ++i) {
    validLoadTileA[i] =
        pA.isRowValid(i * tileGlobalIntervalA) && pA.isColValid(0);
    bufferA[i] =
        validLoadTileA[i] ? pA(i * tileGlobalIntervalA, 0) : float4Zero;
  }

#pragma unroll
  for (unsigned i = 0; i < tileIterationsB; ++i) {
    validLoadTileB[i] =
        pB.isColValid(0) && pB.isRowValid(i * tileGlobalIntervalB);
    bufferB[i] =
        validLoadTileB[i] ? pB(i * tileGlobalIntervalB, 0) : float4Zero;
  }

  aduc::float4 c[tileComputeIterationsA * layoutThread::M]
                     [tileComputeIterationsB * layoutThreadT::N];
  memset(c, 0, sizeof(c));
  bool writeStageIdx = false;
#pragma unroll
  for (unsigned i = 0; i < tileIterationsA; ++i) {
#pragma unroll
    for (unsigned j = 0; j < layoutThread::M; ++j) {
      tileA[writeStageIdx][kInTileA * ratio + j]
           [(i * tileGlobalIntervalA + mInTileA) / ratio]
           [(i * tileGlobalIntervalA + mInTileA) % ratio] = bufferA[i][j];
    }
  }

#pragma unroll
  for (unsigned i = 0; i < tileIterationsB; ++i) {
    tileB[writeStageIdx][kinTileB + i * tileGlobalIntervalB][nInTileB] =
        bufferB[i];
  }

  writeStageIdx = !writeStageIdx;

  __syncthreads();

  aduc::float4 fragmentA[2][tileComputeIterationsA * layoutThreadT::M];
  aduc::float4 fragmentB[2][tileComputeIterationsB * layoutThreadT::N];

#pragma unroll
  for (unsigned i = 0; i < tileComputeIterationsA; ++i) {
    fragmentA[0][i] =
        tileA[!writeStageIdx][0][i * tileSharedIntervalAT + mInTileC];
  }
#pragma unroll
  for (unsigned i = 0; i < tileComputeIterationsB; ++i) {
    fragmentB[0][i] =
        tileB[!writeStageIdx][0][i * tileSharedIntervalBT + nInTileC];
  }

  for (unsigned i = layoutTile::K; i < K + layoutTile::K; i += layoutTile::K) {
    pA.addOffset(0, layoutTileT::K);
    pB.addOffset(layoutTile::K, 0);
#pragma unroll
    for (unsigned j = 0; j < tileIterationsA; ++j) {
      validLoadTileA[j] &= pA.isColValid(0);
      bufferA[j] =
          validLoadTileA[j] ? pA(j * tileGlobalIntervalA, 0) : float4Zero;
    }

#pragma unroll
    for (unsigned j = 0; j < tileIterationsB; ++j) {
      validLoadTileB[j] &= pB.isRowValid(j * tileGlobalIntervalB);
      bufferB[j] =
          validLoadTileB[j] ? pB(j * tileGlobalIntervalB, 0) : float4Zero;
    }

#pragma unroll
    for (unsigned j = 0; j < layoutTile::K; j++) {
      if ((i < K) && (j == layoutTile::K - 1)) {
#pragma unroll
        for (unsigned d = 0; d < tileIterationsA; ++d) {
#pragma unroll
          for (unsigned e = 0; e < layoutThread::M; ++e) {
            tileA[writeStageIdx][kInTileA * ratio + e]
                 [(d * tileGlobalIntervalA + mInTileA) / ratio]
                 [(d * tileGlobalIntervalA + mInTileA) % ratio] = bufferA[d][e];
          }
        }
#pragma unroll
        for (unsigned a = 0; a < tileIterationsB; ++a) {
          tileB[writeStageIdx][kinTileB + a * tileGlobalIntervalB][nInTileB] =
              bufferB[a];
        }
        writeStageIdx = !writeStageIdx;
        __syncthreads();
      }
#pragma unroll
      for (unsigned a = 0; a < tileComputeIterationsA; ++a) {
        fragmentA[(j + 1) % 2][a] =
            tileA[!writeStageIdx][(j + 1) % layoutTile::K]
                 [a * tileSharedIntervalAT + mInTileC];
      }
#pragma unroll
      for (unsigned a = 0; a < tileComputeIterationsB; ++a) {
        fragmentB[(j + 1) % 2][a] =
            tileB[!writeStageIdx][(j + 1) % layoutTile::K]
                 [a * tileSharedIntervalBT + nInTileC];
      }
#pragma unroll
      for (unsigned d = 0; d < tileComputeIterationsA * layoutThread::M; ++d) {
#pragma unroll
        for (unsigned e = 0; e < tileComputeIterationsB * layoutThreadT::N;
             ++e) {
          c[d][e] =
              c[d][e] +
              fragmentB[j % 2][e] *
                  fragmentA[j % 2][d / layoutThread::M][d % layoutThread::M];
        }
      }
    }
  }

#pragma unroll
  for (auto &a : c) {
#pragma unroll
    for (auto &b : a) {
      b = b * alpha;
    }
  }

#pragma unroll
  for (unsigned i = 0; i < tileComputeIterationsA; ++i) {
#pragma unroll
    for (unsigned a = 0; a < layoutThread::M; a++) {
      const bool mValid = pC.isRowValid(a);
#pragma unroll
      for (unsigned b = 0; b < tileComputeIterationsB; b++) {
        const bool nValid = pC.isColValid(b * tileSharedIntervalBT);
        if (mValid && nValid) {
          aduc::float4 result{c[a + i * layoutThread::M][b]};
          if (beta != 0) {
            result = result + pC(a, b * tileSharedIntervalBT) * beta;
          }
          pC(a, b * tileSharedIntervalBT) = result;
        }
      }
    }
    pC.addOffset(tileSharedIntervalAT * ratio, 0);
  }
}
}  // namespace

void r1_HideGmemLatency(const float *deviceAPtr, const float *deviceBPtr,
               float *deviceCPtr, float alpha, float beta, unsigned M,
               unsigned N, unsigned K) {
  using layoutTile = aduc::layout<128, 128, 16>;
  using layoutBlock = aduc::layout<16, 16>;
  using layoutThread = aduc::layout<4, 4>;

  dim3 block(layoutBlock::M * layoutBlock::N);
  dim3 grid((M - 1) / layoutTile::M + 1, (N - 1) / layoutTile::N + 1);

  gemmKernel<layoutTile, layoutBlock, layoutThread><<<grid, block>>>(
      deviceAPtr, deviceBPtr, deviceCPtr, alpha, beta, M, N, K);
}