#include "float4.cuh"
#include <assert.h>
#include "reduce.cuh"

/*
class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = Parameter(init.ones(1, dim, dtype=dtype, device=device))
        self.bias   = Parameter(init.zeros(1, dim, dtype=dtype, device=device))

    def forward(self, x: Tensor) -> Tensor:
        batch = x.shape[0]
        classes = x.shape[1]

        ex = ops.broadcast_to(
                ops.reshape(
                    ops.summation(x, axes=1) / classes, 
                    (-1,1)
                ), x.shape)
        
        varx = ops.broadcast_to(
                ops.reshape(
                    ops.summation((x - ex) ** 2, axes=1) / classes, 
                    (-1,1)),
                x.shape) + self.eps
        
        return ops.broadcast_to(self.weight, x.shape) * \
                (x - ex) / (varx ** 0.5) + ops.broadcast_to(self.bias, x.shape)
*/

constexpr int thread_per_block = 128;

namespace {
	
	__global__
	void kernel(float* x, float* y, float* weight, float* bias, float eps, int N){
		int tid = threadIdx.x;

		f4* x4p = reinterpret_cast<f4*>(&(x[blockIdx.x * N]));
		if(tid >= N / 4) return;
		
		f4 x4 = x4p[tid];
		float sum = 0;
		#pragma unroll
		for(int i = 0; i < 4; i ++){
			sum += x4[i];
		}

		sum = block_reduce_sum<thread_per_block>(sum);
		__shared__ float shared_mean;
		if(tid == 0) shared_mean = sum / N;
		__syncthreads();

		float var = 0;
		#pragma unroll
		for(int i = 0; i < 4; i ++){
			x4[i] = x4[i] - shared_mean;
			var += x4[i] * x4[i];
		}
		var = block_reduce_sum<thread_per_block>(var);
		__shared__ float shared_var;
		if(tid == 0) shared_var = /*1/sqrt*/rsqrtf(var / N + eps);
		__syncthreads();

		f4* y4p = reinterpret_cast<f4*>(&(y[blockIdx.x * N]));
		f4 w4 = reinterpret_cast<f4*>(weight)[tid];
		f4 b4 = reinterpret_cast<f4*>(bias)[tid];

		#pragma unroll
		for(int i = 0; i < 4; i ++){
			x4[i] = x4[i] * w4[i] * shared_var + b4[i];
		}
		y4p[tid] = x4;
	}
}

void layernorm1d(float* x, float* y, float* weight, float* bias, float eps, int batch, int N){
	assert(N % 4 == 0 && N <= thread_per_block * 4);
	// 一个block处理一行
	kernel<<<batch, thread_per_block>>>(x, y, weight, bias, eps, N);
}

