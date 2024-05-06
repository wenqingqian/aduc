#include <cuda_runtime.h>
#include <iostream>
#include "matrix.h"
#include "debug.cuh"
#include <omp.h>

namespace aduc{

struct __device_builtin__ __builtin_align__(16) float4{
	float data_[4];

	__host__ __device__ float &operator[](unsigned idx){
		return data_[idx]; 
	}

	__host__ __device__ float operator[](unsigned idx) const {
		return data_[idx];
	}

	__host__ __device__ float4 operator+(const float4 &a){
		return float4{data_[0]+a.data_[0], data_[1]+a.data_[1],
						data_[2]+a.data_[2], data_[3]+a.data_[3]};
	}

	__host__ __device__ float4 operator*(float f) const{
		return float4{data_[0]*f, data_[1]*f, data_[2]*f, data_[3]*f};
	}

	__host__ __device__ float operator*(const float4 &f) const{
		return data_[0]*f[0] + data_[1]*f[1] + data_[2]*f[2] + data_[3]*f[3];
	}
};

template < int warp_size = 32, class T >
__device__ T warp_reduce_sum(T val) {
	#pragma unroll
	for (int mask = warp_size >> 1; mask >= 1; mask >>= 1) {
		val += __shfl_xor_sync(0xffffffff, val, mask);
	}
	return val;
}

__global__
void dot_product(float* x, float* y, float* out, int N){
	// N 必须是128的倍数
	// 一个warp负责128个数

	int line = blockIdx.x * 128 / 4 + threadIdx.x;
	aduc::float4 x4 = reinterpret_cast<aduc::float4 *>(x)[line];
	aduc::float4 y4 = reinterpret_cast<aduc::float4 *>(y)[line];

	float res = x4 * y4;
	res = warp_reduce_sum(res);

	if(threadIdx.x == 0){
		atomicAdd(out, res);
	}
}
}

int main(){
	omp_set_num_threads(omp_get_num_procs());

	constexpr int N = 128;
	matrix<float> x(1, N);
	matrix<float> y(N, 1);

	float* x_d, *y_d, *out;
	
	EcudaMalloc(&x_d, N * sizeof(float));
	EcudaMalloc(&y_d, N * sizeof(float));
	EcudaMalloc(&out, sizeof(float));

	EcudaMemcpy(x_d, x.unsafe_data(), N * sizeof(float), cudaMemcpyHostToDevice);
	EcudaMemcpy(y_d, y.unsafe_data(), N * sizeof(float), cudaMemcpyHostToDevice);

	matrix<float> r = x * y;
	x.show();
	y.show();
	r.show();

	aduc::dot_product<<<N/128,32>>>(x_d, y_d, out, N);

	cudaDeviceSynchronize();

	float res;
	EcudaMemcpy(&res, out, sizeof(float), cudaMemcpyDeviceToHost);
	
	printf("res: %f, error: %f\n", res, res - r[0]);
}