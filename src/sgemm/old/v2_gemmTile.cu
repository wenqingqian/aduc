#include "util.cuh"

template < class layoutBlock, class layoutThread >
__global__ void gemmTileKernel(const float * __restrict__ A, const float * __restrict__ B, float *  __restrict__ C,
	float alpha, float beta, unsigned M, unsigned N, unsigned K) 
{
	aduc::tensor2d<const float> tsA(A, M, K);
	aduc::tensor2d<const aduc::float4> tsB(B, K, N / layoutThread::M);
	aduc::tensor2d<aduc::float4> tsC(C, M, N / layoutThread::M);

	unsigned offsetX = (blockIdx.x * layoutBlock::M + threadIdx.x) * layoutThread::M;
	unsigned offsetY = blockIdx.y * layoutBlock::N + threadIdx.y;
	// A 矩阵按行划分, 偏移到当前线程需要处理位置
	tsA.addOffset(offsetX, 0);

	// B 矩阵按列划分, 偏移到当前线程需要处理位置, 列以float4为单位(减少访存指令, 缓存行)
	tsB.addOffset(0, offsetY);

	tsC.addOffset(offsetX, offsetY);

	if ( !tsC.isValid() ){
		return;
	}

	aduc::float4 result[4];
	memset(result, 0, sizeof(aduc::float4)*4);

	for ( int i = 0; i < K; i ++ ){
		aduc::float4 tmpB = tsB(i, 0);
		#pragma unroll
		for ( int j = 0; j < 4; j ++ ){
			result[j] = result[j] + tmpB * tsA(j, i);
		}
	}

	#pragma unroll
	for ( int i = 0; i < 4; i ++ ){
		tsC(i, 0) = result[i] * alpha + tsC(i, 0) * beta;
	}
}

def_gemm(gemmTile)
{
	using layoutBlock  = aduc::layout<16, 16>;
	// 一个block 16x16 线程
	using layoutThread = aduc::layout<4, 4>;
	// 一个线程 4x4 数据
	dim3 block(layoutBlock::M, layoutBlock::N);
	dim3 grid((M / layoutThread::M - 1) / block.x + 1, (N / layoutThread::N - 1) / block.y + 1);
	gemmTileKernel<layoutBlock, layoutThread><<<grid, block>>>(A, B, C, alpha, beta, M, N, K);
}