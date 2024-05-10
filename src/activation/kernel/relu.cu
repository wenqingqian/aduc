#include <assert.h>
#include "float4.cuh"
namespace {
	__global__
	void kernel(float* x, float* y, int N){
		int tid = blockIdx.x * blockDim.x + threadIdx.x;
		
		if(tid >= N / 4) return;
		
		f4 x4 = reinterpret_cast<f4*>(x)[tid];

		#pragma unroll
		for(int i = 0; i < 4; i ++){
			x4[i] = fmaxf(0, x4[i]);
		}

		f4* out = reinterpret_cast<f4*>(y);
		out[tid] = x4;
	}
}

void relu(float* x, float* y, int N){
	assert(N % 4 == 0);
	kernel<<<(N + 127) / 128, 32>>>(x, y, N);
}