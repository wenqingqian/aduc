#include "util.cuh"
#include <cuda_runtime.h>

// https://github.com/openmlsys/openmlsys-cuda/tree/main

template < class layoutTile, class layoutBlock, class layoutThread >
__global__ void gemmShareMemKernel(const float * __restrict__ A, const float * __restrict__ B, float *  __restrict__ C,
	float alpha, float beta, unsigned M, unsigned N, unsigned K) 
{
	// 便捷起见, 不用复杂的变量来表示
	aduc::tensor2d<const aduc::float4> tsA(A, M, K / 4);
	aduc::tensor2d<const aduc::float4> tsB(B, K, N / 4);

	const unsigned UthreadIdx = threadIdx.y * 16 + threadIdx.x;

	tsA.addOffset(128 * blockIdx.x + UthreadIdx / 4, UthreadIdx % 4);
	// tsB 我是竖着切, openmlsys好像也是横着切
	tsB.addOffset(UthreadIdx % 16, blockIdx.y * 32 + UthreadIdx / 16);

	aduc::tensor2d<aduc::float4> tsC(C, M, N / 4);
	tsC.addOffset(128 * blockIdx.x + threadIdx.y * 4, 32 * blockIdx.y + threadIdx.x);

	if ( !tsC.isValid() )
		return;

	constexpr aduc::float4 float4Zero{0.f,0.f,0.f,0.f};

	__shared__ aduc::float4 tileA[128][4];
	__shared__ aduc::float4 tileB[16][32];

	// store the global mem and move to share mem
	aduc::float4 bufferA[2];
	aduc::float4 bufferB[2];

	// 4 float4 per thread, check these 4 f4 validation
	// tsA's col and tsB's row are variable, so for A: check the row here and col in loops
	bool validA[2];
	bool validB[2];
	#pragma unroll
	for ( int i = 0; i < 2; i ++ ){
		validA[i] = tsA.isRowValid(i * 64);
		validB[i] = tsB.isColValid(i * 16);
	}

	// buffer: the IO-time layout, fragment: compute-time layout
	aduc::float4 fragmentA[2];
	aduc::float4 fragmentB[2];

	// 每个线程计算的值
	aduc::float4 result[8][2];
	memset(result, 0, sizeof(result));

	for ( int i = 0; i < K / 16; i ++ ){
	// 使用 for ( int i = 0; i < K; i += 16 ) 会慢1%
		#pragma unroll
		for ( int j = 0; j < 2; j ++ ){
			validA[j] &= tsA.isColValid();
			bufferA[j] = validA[j] ? tsA(j * 64, 0) : float4Zero;
		}
		#pragma unroll
		for ( int j = 0; j < 2; j ++ ){
			validB[j] &= tsB.isRowValid();
			bufferB[j] = validB[j] ? tsB(0, j * 16) : float4Zero;
		}
		__syncthreads();

		#pragma unroll
		for ( int j = 0; j < 2; j ++ ){
			tileA[threadIdx.y * 4 + threadIdx.x / 4 + j * 64][threadIdx.x % 4] = bufferA[j];
		}
		#pragma unroll
		for ( int j = 0; j < 2; j ++ ){
			tileB[threadIdx.x][threadIdx.y + j * 16] = bufferB[j];
		}
		__syncthreads();

		#pragma unroll
		for ( int j = 0; j < 16; j ++ ){
			#pragma unroll
			for ( int j2 = 0; j2 < 2; j2 ++ ){
				#pragma unroll
				for ( int j3 = 0; j3 < 4; j3 ++ ){
					fragmentA[j2][j3] = tileA[j2 * 64 + threadIdx.y * 4 + j3][j / 4][j % 4];
				}
			}
			#pragma unroll
			for ( int j2 = 0; j2 < 2; j2 ++ ){
				fragmentB[j2] = tileB[j][threadIdx.x + j2 * 16];
			}
			
			#pragma unroll
			for( int j2 = 0; j2 < 8; j2 ++ ){
				#pragma unroll
				for ( int j3 = 0; j3 < 2; j3 ++ ){
					// 在这里直接计算alpha会更慢
					result[j2][j3] = result[j2][j3] + fragmentB[j3] * fragmentA[j2 / 4][j2 % 4];
				}
			}
		}
		tsA.addOffset(0, 4);
		tsB.addOffset(16, 0);
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

def_gemm(gemmShareMem)
{
	using layoutTile = aduc::layout<128, 128, 16>;
	// 一个线程块处理 128x128数据, 每个线程在v2的基础上翻一倍, 将A的列和B的行分段(16个一段)放进共享内存
	using layoutBlock  = aduc::layout<16, 16>;
	// 一个block 16x16 线程
	using layoutThread = aduc::layout<4, 4>;
	// 一个线程 4x4 数据
	dim3 block(layoutBlock::M, layoutBlock::N);
	dim3 grid((M - 1) / layoutTile::M + 1, (N - 1) / layoutTile::N + 1);
	gemmShareMemKernel<layoutTile, layoutBlock, layoutThread><<<grid, block>>>(A, B, C, alpha, beta, M, N, K);
}