#include <cuda_runtime.h>
#include "util.cuh"
// 166ms

__global__ void gemmNaiveKernel(const float * A,const float * B, float * C,
	float alpha, float beta, unsigned M, unsigned N,unsigned K) 
{   
	unsigned int m = threadIdx.x + blockDim.x * blockIdx.x;
	unsigned int n = threadIdx.y + blockDim.y * blockIdx.y;
	if (m >= M || n >= N)
		return;
	float c = 0;
	for (unsigned k = 0; k < K; ++k) {
		c += A[m * K + k] * B[k * N + n];
	}
	c = c * alpha;
	float result = c;
	if (beta != 0) {
		result = result + C[m * N + n] * beta;
	}
	C[m * N + n] = result;
}


def_gemm(gemmNaive)
{
	dim3 block(16, 16);
	dim3 grid((M - 1) / block.x + 1, (N - 1) / block.y + 1);

	gemmNaiveKernel<<<grid, block>>>(A, B, C, alpha, beta, M, N, K);
}