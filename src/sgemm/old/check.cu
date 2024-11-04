#include "util.cuh"

namespace {
template <typename layoutTile, typename layoutBlock, typename layoutThread>
__global__ void gemmKernel(const float *__restrict__ A,
                           const float *__restrict__ B, float *__restrict__ C,
                           float alpha, float beta, unsigned M, unsigned N,
                           unsigned K) {

  using layoutTileT =
      aduc::layout<32, 32, 4>;
  using layoutThreadT =
      aduc::layout<1, 1>;

  constexpr aduc::float4 float4Zero{0.f, 0.f, 0.f, 0.f};

  __shared__ aduc::float4 tileA[2][16][32];
  __shared__ aduc::float4 tileB[2][16][32];

  aduc::tensor2d<const aduc::float4> pA{A, M, K / 4};
  pA.addOffset(128 * blockIdx.y + threadIdx.x / 4, threadIdx.x % 4);
  aduc::tensor2d<const aduc::float4> pB{B, K, N / 4};
  pB.addOffset(threadIdx.x / 32,
               32 * blockIdx.x + threadIdx.x % 32 * 1);
  aduc::tensor2d<aduc::float4> pC{C, M, N / 4};
  pC.addOffset(128 * blockIdx.y + threadIdx.x / 16 * 4,
               32 * blockIdx.x + threadIdx.x % 16 * 1);

  aduc::float4 bufferA[2];
  aduc::float4 bufferB[2];
  bool validLoadTileA[2];
  bool validLoadTileB[2];

#pragma unroll
  for (unsigned i = 0; i < 2; ++i) {
    validLoadTileA[i] =
        pA.isRowValid(i * 64) && pA.isColValid(0);
    bufferA[i] =
        validLoadTileA[i] ? pA(i * 64, 0) : float4Zero;
  }

#pragma unroll
  for (unsigned i = 0; i < 2; ++i) {
    validLoadTileB[i] =
        pB.isColValid(0) && pB.isRowValid(i * 8);
    bufferB[i] =
        validLoadTileB[i] ? pB(i * 8, 0) : float4Zero;
  }

  aduc::float4 c[2 * 4][2 * 1];
  memset(c, 0, sizeof(c));
  bool writeStageIdx = false;
#pragma unroll
  for (unsigned i = 0; i < 2; ++i) {
#pragma unroll
    for (unsigned j = 0; j < 4; ++j) {
      tileA[writeStageIdx][threadIdx.x % 4 * 4 + j]
           [(i * 64 + threadIdx.x / 4) / 4]
           [(i * 64 + threadIdx.x / 4) % 4] = bufferA[i][j];
    }
  }

#pragma unroll
  for (unsigned i = 0; i < 2; ++i) {
    tileB[writeStageIdx][threadIdx.x / 32 + i * 8][threadIdx.x % 32] =
        bufferB[i];
  }

  writeStageIdx = !writeStageIdx;

  __syncthreads();

  aduc::float4 fragmentA[2][2 * 1];
  aduc::float4 fragmentB[2][2 * 1];

#pragma unroll
  for (unsigned i = 0; i < 2; ++i) {
    fragmentA[0][i] =
        tileA[!writeStageIdx][0][i * 16 + threadIdx.x / 16];
  }
#pragma unroll
  for (unsigned i = 0; i < 2; ++i) {
    fragmentB[0][i] =
        tileB[!writeStageIdx][0][i * 16 + threadIdx.x % 16];
  }

  for (unsigned i = 16; i < K + 16; i += 16) {
    pA.addOffset(0, 4);
    pB.addOffset(16, 0);
#pragma unroll
    for (unsigned j = 0; j < 2; ++j) {
      validLoadTileA[j] &= pA.isColValid(0);
      bufferA[j] =
          validLoadTileA[j] ? pA(j * 64, 0) : float4Zero;
    }

#pragma unroll
    for (unsigned j = 0; j < 2; ++j) {
      validLoadTileB[j] &= pB.isRowValid(j * 8);
      bufferB[j] =
          validLoadTileB[j] ? pB(j * 8, 0) : float4Zero;
    }

#pragma unroll
    for (unsigned j = 0; j < 16; j++) {
      if ((i < K) && (j == 16 - 1)) {
#pragma unroll
        for (unsigned d = 0; d < 2; ++d) {
#pragma unroll
          for (unsigned e = 0; e < 4; ++e) {
            tileA[writeStageIdx][threadIdx.x % 4 * 4 + e]
                 [(d * 64 + threadIdx.x / 4) / 4]
                 [(d * 64 + threadIdx.x / 4) % 4] = bufferA[d][e];
          }
        }
#pragma unroll
        for (unsigned a = 0; a < 2; ++a) {
          tileB[writeStageIdx][threadIdx.x / 32 + a * 8][threadIdx.x % 32] =
              bufferB[a];
        }
        writeStageIdx = !writeStageIdx;
        __syncthreads();
      }
#pragma unroll
      for (unsigned a = 0; a < 2; ++a) {
        fragmentA[(j + 1) % 2][a] =
            tileA[!writeStageIdx][(j + 1) % 16]
                 [a * 16 + threadIdx.x / 16];
      }
#pragma unroll
      for (unsigned a = 0; a < 2; ++a) {
        fragmentB[(j + 1) % 2][a] =
            tileB[!writeStageIdx][(j + 1) % 16]
                 [a * 16 + threadIdx.x % 16];
      }
#pragma unroll
      for (unsigned d = 0; d < 2 * 4; ++d) {
#pragma unroll
        for (unsigned e = 0; e < 2 * 1;
             ++e) {
          c[d][e] =
              c[d][e] +
              fragmentB[j % 2][e] *
                  fragmentA[j % 2][d / 4][d % 4];
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
  for (unsigned i = 0; i < 2; ++i) {
#pragma unroll
    for (unsigned a = 0; a < 4; a++) {
      const bool mValid = pC.isRowValid(a);
#pragma unroll
      for (unsigned b = 0; b < 2; b++) {
        const bool nValid = pC.isColValid(b * 16);
        if (mValid && nValid) {
          aduc::float4 result{c[a + i * 4][b]};
          if (beta != 0) {
            result = result + pC(a, b * 16) * beta;
          }
          pC(a, b * 16) = result;
        }
      }
    }
    pC.addOffset(16 * 4, 0);
  }
}
}  // namespace

void r1_HideGmemLatency(const float *deviceAPtr, const float *deviceBPtr,
               float *deviceCPtr, float alpha, float beta, unsigned M,
               unsigned N, unsigned K) {
  using layoutTile = aduc::layout<128, 128, 16>;
  using layoutBlock = aduc::layout<16, 16>;
  using layoutThread = aduc::layout<4, 4>;

  dim3 block(16 * layoutBlock::N);
  dim3 grid((M - 1) / 128 + 1, (N - 1) / 128 + 1);

  gemmKernel<layoutTile, layoutBlock, layoutThread><<<grid, block>>>(
      deviceAPtr, deviceBPtr, deviceCPtr, alpha, beta, M, N, K);
}