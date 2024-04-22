#include "util.cuh"
#include <cuda_runtime.h>


template < class layoutTile, class layoutBlock, class layoutThread >
__global__ void gemmHideGmemLatencyKernel(const float * __restrict__ A, const float * __restrict__ B, float *  __restrict__ C,
	float alpha, float beta, unsigned M, unsigned N, unsigned K) 
{
	// 便捷起见, 不用复杂的变量来表示
	aduc::tensor2d<const aduc::float4> tsA(A, M, K / 4);
	aduc::tensor2d<const aduc::float4> tsB(B, K, N / 4);

	const unsigned UthreadIdx = threadIdx.y * 16 + threadIdx.x;

	tsA.addOffset(128 * blockIdx.x + UthreadIdx / 4, UthreadIdx % 4);
	// tsB 横着切
	tsB.addOffset(UthreadIdx / 32, blockIdx.y * 32 + UthreadIdx % 32);

	aduc::tensor2d<aduc::float4> tsC(C, M, N / 4);
	tsC.addOffset(128 * blockIdx.x + threadIdx.y * 4, 32 * blockIdx.y + threadIdx.x);

	if ( !tsC.isValid() )
		return;

	constexpr aduc::float4 float4Zero{0.f,0.f,0.f,0.f};

	// __shared__ aduc::float4 tileA[128][4];
	__shared__ aduc::float4 tileA[2][16][32];
	__shared__ aduc::float4 tileB[2][16][32];

	// store the global mem and move to share mem
	aduc::float4 bufferA[2];
	aduc::float4 bufferB[2];

	bool validA[2];
	bool validB[2];

	// buffer: the IO-time layout, fragment: compute-time layout
	aduc::float4 fragmentA[2][2];
	aduc::float4 fragmentB[2][2];

	// 每个线程计算的值
	aduc::float4 result[8][2];

	#pragma unroll
	for ( int i = 0; i < 2; i ++ ){
		validA[i]  = tsA.isRowValid(i * 64) & tsA.isColValid();
		bufferA[i] = validA[i] ? tsA(i * 64, 0) : float4Zero;
	}
	#pragma unroll
	for ( int i = 0; i < 2; i ++ ){
		validB[i]  = tsB.isColValid() & tsB.isRowValid(i * 8);
		bufferB[i] = validB[i] ? tsB(i * 8, 0) : float4Zero;
	}

	memset(result, 0, sizeof(result));
	__syncthreads();

	#pragma unroll
	for ( int j = 0; j < 2; j ++ ){
		#pragma unroll
		for ( int k = 0; k < 4; k ++ ){
			tileA[0][(threadIdx.x % 4) * 4 + k][threadIdx.y + j * 16][threadIdx.x / 4] = bufferA[j][k];
		}
	}
	#pragma unroll
	for ( int j = 0; j < 2; j ++ ){
		tileB[0][UthreadIdx / 32 + j * 8][UthreadIdx % 32] = bufferB[j];
	}
	__syncthreads();

	#pragma unroll
	for ( int j = 0; j < 2; j ++ ){
		fragmentA[0][j] = tileA[0][0][threadIdx.y + j * 16];
	}
	#pragma unroll
	for ( int j = 0; j < 2; j ++ ){
		fragmentB[0][j] = tileB[0][0][threadIdx.x + j * 16];
	}

	bool tileIdx = 1;
	
	for ( int i = 0; i < K / 16; i ++ ){
	// 使用 for ( int i = 0; i < K; i += 16 ) 会慢1%
		tsA.addOffset(0, 4);
		tsB.addOffset(16, 0);

		#pragma unroll
		for ( int j = 0; j < 2; j ++ ){
			validA[j] &= tsA.isColValid();
			bufferA[j] = validA[j] ? tsA(j * 64, 0) : float4Zero;
		}
		#pragma unroll
		for ( int j = 0; j < 2; j ++ ){
			validB[j] &= tsB.isRowValid(j * 8);
			bufferB[j] = validB[j] ? tsB(j * 8, 0) : float4Zero;
		}

		#pragma unroll
		for ( int j = 0; j < 15; j ++ ){
			// inner loop, next fragment
			// 这里加载到0则计算用1, 这里为1则计算用0, 不断循环每次都是加载下次数据,用上次数据
			#pragma unroll
			for ( int j2 = 0; j2 < 2; j2 ++ ){
				fragmentA[(j + 1) % 2][j2] = tileA[!tileIdx][j + 1][threadIdx.y + j2 * 16];
			}
			#pragma unroll
			for ( int j2 = 0; j2 < 2; j2 ++ ){
				fragmentB[(j + 1) % 2][j2] = tileB[!tileIdx][j + 1][threadIdx.x + j2 * 16];
			}
			
			#pragma unroll
			for( int j2 = 0; j2 < 8; j2 ++ ){
				#pragma unroll
				for ( int j3 = 0; j3 < 2; j3 ++ ){
					// 在这里直接计算alpha会更慢
					result[j2][j3] = result[j2][j3] + fragmentB[j % 2][j3] * fragmentA[j % 2][j2 / 4][j2 % 4];
				}
			}
		}

		if ( i < K / 16 - 1 ){
			#pragma unroll
			for ( int j = 0; j < 2; j ++ ){
				#pragma unroll
				for ( int k = 0; k < 4; k ++ ){
					tileA[tileIdx][(threadIdx.x % 4) * 4 + k][threadIdx.y + j * 16][threadIdx.x / 4] = bufferA[j][k];
				}
			}
			#pragma unroll
			for ( int j = 0; j < 2; j ++ ){
				tileB[tileIdx][UthreadIdx / 32 + j * 8][UthreadIdx % 32] = bufferB[j];
			}
			tileIdx = !tileIdx;
			__syncthreads();
		}

		// outer loop
		#pragma unroll
		for ( int j = 0; j < 2; j ++ ){
			fragmentA[0][j] = tileA[!tileIdx][0][threadIdx.y + j * 16];
		}
		#pragma unroll
		for ( int j = 0; j < 2; j ++ ){
			fragmentB[0][j] = tileB[!tileIdx][0][threadIdx.x + j * 16];
		}

		#pragma unroll
		for( int j2 = 0; j2 < 8; j2 ++ ){
			#pragma unroll
			for ( int j3 = 0; j3 < 2; j3 ++ ){
				// 在这里直接计算alpha会更慢
				result[j2][j3] = result[j2][j3] + fragmentB[1][j3] * fragmentA[1][j2 / 4][j2 % 4];
			}
		}

	}

	#pragma unroll
	for ( int i = 0; i < 8; i ++ ){
		#pragma unroll
		for ( int j = 0; j < 2; j ++ ){
			result[i][j] = result[i][j] * alpha;
		}
	}

	// 四个分散result矩阵, 按
	// 1 2
	// 3 4  顺序计算
	#pragma unroll
	for ( int i = 0; i < 2; i ++ ){ // 分离12/34
		
		#pragma unroll
		for ( int j = 0; j < 4; j ++ ){ // result矩阵(4x(1*f4)) x2 按行拆开
			bool isRowValid = tsC.isRowValid(j);
			#pragma unroll
			for ( int k = 0; k < 2; k ++ ){ // 分离 1/2/3/4
				bool isCowValid = tsC.isColValid(k * 16);
				if( isCowValid && isRowValid ){
					tsC(j, k * 16) = tsC(j, k * 16) * beta + result[i * 4 + j][k];
				}
			}
		}

		tsC.addOffset(64, 0);
	}

}

def_gemm(gemmHideGmemLatency)
{
	using layoutTile = aduc::layout<128, 128, 16>;
	// 一个线程块处理 128x128数据, 每个线程在v2的基础上翻一倍, 将A的列和B的行分段(16个一段)放进共享内存
	using layoutBlock  = aduc::layout<16, 16>;
	// 一个block 16x16 线程
	using layoutThread = aduc::layout<4, 4>;
	// 一个线程 4x4 数据
	dim3 block(layoutBlock::M, layoutBlock::N);
	dim3 grid((M - 1) / layoutTile::M + 1, (N - 1) / layoutTile::N + 1);
	gemmHideGmemLatencyKernel<layoutTile, layoutBlock, layoutThread><<<grid, block>>>(A, B, C, alpha, beta, M, N, K);
}