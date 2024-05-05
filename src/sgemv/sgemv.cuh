#pragma once
#include <cuda_runtime.h>

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
void sgemv_eq32_kernel(float* __restrict__ A, float* __restrict__ x, float* __restrict__ y, 
			const int M, const int N)
{
	int warp = threadIdx.x / 32; // 0 - 3
	int lane = threadIdx.x % 32;

	// The line number for this warp
	int line = blockIdx.x * 4 + warp;

	if(line >= M) return;

	float res = A[line * 32 + lane] * x[lane];
	
	res = warp_reduce_sum(res);

	if(lane == 0){
		y[line] = res;
	}
}


template < int thread_per_row >
__global__
void sgemv_lt32_kernel(float* __restrict__ A, float* __restrict__ x, float* __restrict__ y, 
			const int M, const int N)
{
	int warp = threadIdx.x / 32; // 0 - 3
	int lane = threadIdx.x % 32;

	// 一个warp负责多行, warp中线程在每行中的索引

	int kwarp = lane / thread_per_row;
	int klane = lane % thread_per_row;

	constexpr int row_per_warp = 32 / thread_per_row;
	// The line number for this warp
	int line = blockIdx.x * 4 * row_per_warp + warp * row_per_warp + kwarp;

	if(line >= M) return;

	float res = (klane < N) ? A[line * N + klane] * x[klane] : 0;
	
	res = warp_reduce_sum<thread_per_row>(res);

	if(klane == 0){
		y[line] = res;
	}
}

template < int thread_per_row, int float4_per_row >
__global__
void sgemv_gt32_kernel(float* __restrict__ A, float* __restrict__ x, float* __restrict__ y, 
			const int M, const int N)
{
	// 此函数只处理N小于等于128并且N是4的倍数

	constexpr int row_per_warp = 32 / thread_per_row;

	int warp = threadIdx.x / 32; // 0 - 3
	int lane = threadIdx.x % 32;

	int kwarp = lane / thread_per_row;
	int klane = lane % thread_per_row;

	int line = blockIdx.x * 4 * row_per_warp + warp * row_per_warp + kwarp;

	if(line >= M) return;


	aduc::float4 A4, x4;
	if(klane < float4_per_row){
		A4 = reinterpret_cast<aduc::float4 *>(A)[line * N / 4 + klane];
		x4 = reinterpret_cast<aduc::float4 *>(x)[klane];
	}else{
		A4 = aduc::float4{0,0,0,0};
		x4 = aduc::float4{0,0,0,0};
	}

	float res = A4 * x4;
	res = warp_reduce_sum<thread_per_row>(res);

	if(klane == 0){
		y[line] = res;
	}
}

}

inline constexpr int align2powerHelper(int N, int C){
	if ( N < C ){
		return align2powerHelper(N << 1, C);
	}
	return N;
}

inline constexpr int align2power(int n){
	if ( (n & (n - 1)) == 0 )
		return n;
	return align2powerHelper(1, n);
}

inline void sgemv_eq32(float* A, float* x, float* y, 
			const int M, const int N)
{
	// one warp perform one line, one block perform four line
	aduc::sgemv_eq32_kernel<<<M / 4 , 128>>>(A,x,y,M,N);
}

template < int N >
void sgemv_lt32(float* A, float* x, float* y, 
			const int M, const int N_unuse)
{
	// 把 N 对齐到 2^n 上
	aduc::sgemv_lt32_kernel<align2power(N)><<<M / (128 / align2power(N)),128>>>(A,x,y,M,N);
}


template < int N >
void sgemv_gt32(float* A, float* x, float* y, 
			const int M, const int N_unuse)
{
	aduc::sgemv_gt32_kernel<align2power(N) / 4, N / 4><<<M / (128 * 4 / align2power(N)),128>>>(A,x,y,M,N);
}