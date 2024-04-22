#include "util.cuh"

namespace {
template <typename layoutTile, typename layoutBlock, typename layoutThread>
__global__ void r1_ColMajorSMAKernel(const float *__restrict__ A,
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

  __shared__ aduc::float4 tileA[layoutTile::K][layoutTileT::M];
  __shared__ aduc::float4 tileB[layoutTile::K][layoutTileT::N];

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
    validLoadTileA[i] = pA.isRowValid(i * tileGlobalIntervalA);
  }

#pragma unroll
  for (unsigned i = 0; i < tileIterationsB; ++i) {
    validLoadTileB[i] = pB.isColValid(0);
  }

  aduc::float4 c[tileComputeIterationsA * layoutThread::M]
                     [tileComputeIterationsB * layoutThreadT::N];
  memset(c, 0, sizeof(c));

  aduc::float4 fragmentA[tileComputeIterationsA * layoutThreadT::M];
  aduc::float4 fragmentB[tileComputeIterationsB * layoutThreadT::N];

  for (unsigned i = 0; i < K; i += layoutTile::K) {
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

    __syncthreads();
#pragma unroll
    for (unsigned a = 0; a < tileIterationsA; ++a) {
#pragma unroll
      for (unsigned j = 0; j < layoutThread::M; ++j) {
        tileA[kInTileA * ratio + j]
             [(a * tileGlobalIntervalA + mInTileA) / ratio]
             [(a * tileGlobalIntervalA + mInTileA) % ratio] = bufferA[a][j];
      }
    }

#pragma unroll
    for (unsigned a = 0; a < tileIterationsB; ++a) {
      tileB[kinTileB + a * tileGlobalIntervalB][nInTileB] = bufferB[a];
    }
    __syncthreads();

#pragma unroll
    for (unsigned j = 0; j < layoutTile::K; j++) {
#pragma unroll
      for (unsigned a = 0; a < tileComputeIterationsA; ++a) {
        fragmentA[a] = tileA[j][a * tileSharedIntervalAT + mInTileC];
      }
#pragma unroll
      for (unsigned a = 0; a < tileComputeIterationsB; ++a) {
        fragmentB[a] = tileB[j][a * tileSharedIntervalBT + nInTileC];
      }
#pragma unroll
      for (unsigned d = 0; d < tileComputeIterationsA * layoutThread::M; ++d) {
#pragma unroll
        for (unsigned e = 0; e < tileComputeIterationsB * layoutThreadT::N;
             ++e) {
          c[d][e] =
              c[d][e] + fragmentB[e] *
                            fragmentA[d / layoutThread::M][d % layoutThread::M];
        }
      }
    }
    pA.addOffset(0, layoutTileT::K);
    pB.addOffset(layoutTile::K, 0);
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

void r1_ColMajorSMA(const float *deviceAPtr, const float *deviceBPtr,
                       float *deviceCPtr, float alpha, float beta, unsigned M,
                       unsigned N, unsigned K) {
  using layoutTile = aduc::layout<128, 128, 16>;
  using layoutBlock = aduc::layout<16, 16>;
  using layoutThread = aduc::layout<4, 4>;

  dim3 block(layoutBlock::M * layoutBlock::N);
  dim3 grid((M - 1) / layoutTile::M + 1, (N - 1) / layoutTile::N + 1);

  r1_ColMajorSMAKernel<layoutTile, layoutBlock, layoutThread><<<grid, block>>>(
      deviceAPtr, deviceBPtr, deviceCPtr, alpha, beta, M, N, K);
}