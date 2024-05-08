#pragma once

template < int warp_size = 32, class T >
__device__ __inline__
T warp_reduce_sum(T val) {
	#pragma unroll
	for (int mask = warp_size >> 1; mask >= 1; mask >>= 1) {
		val += __shfl_xor_sync(0xffffffff, val, mask);
	}
	return val;
}

template < int warp_size = 32, class T >
__device__ __inline__
T warp_reduce_max(T val) {
	#pragma unroll
	for (int mask = 32 >> 1; mask >= 1; mask >>= 1) {
		val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, mask));
	}
	return val;
}

template < int block_size, class T >
__device__ __inline__
T block_reduce_sum(T val) {
	constexpr int warp_num = (block_size + 32 - 1) / 32;

	int warp = threadIdx.x / 32;
	int lane = threadIdx.x % 32;

	__shared__ T warp_sum[warp_num];

	val = warp_reduce_sum(val);
	if ( lane == 0 ) {
		warp_sum[warp] = val;
	}
	__syncthreads();

	val = warp_reduce_sum( (lane < warp_num) ? warp_sum[lane] : T(0) );

	return val; 
}

template < int block_size, class T >
__device__ __inline__
T block_reduce_max(T val) {
	constexpr int warp_num = (block_size + 32 - 1) / 32;

	int warp = threadIdx.x / 32;
	int lane = threadIdx.x % 32;

	__shared__ T warp_max[warp_num];

	val = warp_reduce_max(val);
	if ( lane == 0 ) {
		warp_max[warp] = val;
	}
	__syncthreads();

	val = warp_reduce_max( (lane < warp_num) ? warp_max[lane] : T(0) );

	return val; 
}
