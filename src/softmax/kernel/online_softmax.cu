#include "float4.cuh"
#include <assert.h>
#include <cfloat>
#include "reduce.cuh"
#include "debug.cuh"
#include "atomicMaxFloat.cuh"

namespace {

struct __align__(8) osm_data {
	float sum;
	float maxn;
};

__device__
void online_softmax_reduce(osm_data& glob, osm_data& tmp){
	bool is_prev_b = glob.maxn > tmp.maxn;
	osm_data& B = is_prev_b ? glob : tmp;
	osm_data& S = is_prev_b ? tmp : glob;

	glob.sum = B.sum + S.sum * expf(S.maxn - B.maxn);
	glob.maxn= B.maxn;
}

template < int N >
__global__ 
void kernel(float* __restrict__ x, float* __restrict__ y, int N_)
{
	// block内4个warp

	// 一维向量的online softmax可以跨block吗
	
	// int lane = threadIdx.x % 32;
	int warp = threadIdx.x / 32;

	// pass 1
	osm_data all_batch { sum:0, maxn:-FLT_MIN };

	aduc::float4 data[N / 512];

	#pragma unroll
	for( int turn = 0; turn < N / 512; ++ turn){
		data[turn] = reinterpret_cast<aduc::float4 *>(x)[threadIdx.x + turn * 128];
		#pragma unroll
		for( int i = 0; i < 4; i ++ ){
			osm_data tmp { sum:1/*e^0*/, maxn:data[turn][i] };
			online_softmax_reduce(all_batch, tmp);
		}
	}

	__shared__ osm_data sm[4];
	#pragma unroll
	for (int mask = 32 >> 1; mask >= 1; mask >>= 1) {
		float sum_tmp = __shfl_xor_sync(0xffffffff, all_batch.sum, mask);
		float max_tmp = __shfl_xor_sync(0xffffffff, all_batch.maxn, mask);
		osm_data tmp {sum:sum_tmp, maxn:max_tmp};
		online_softmax_reduce(all_batch, tmp);
	}
	sm[warp] = all_batch;
	__syncthreads();

	if(threadIdx.x == 0){
		#pragma unroll
		for( int i = 0; i < 3; i ++ ){
			online_softmax_reduce(sm[i+1], sm[i]);
		}
	}
	__syncthreads();

	all_batch = sm[3];

	// pass 2
	aduc::float4 *out = reinterpret_cast<aduc::float4 *>(y);

	#pragma unroll
	for( int turn = 0; turn < N / 512; ++ turn ){
		#pragma unroll
		for( int i = 0; i < 4; i ++ ){
			data[turn][i] = expf(data[turn][i] - all_batch.maxn);
		}
		out[threadIdx.x + turn * 128] = data[turn] / all_batch.sum;
	}
}

}

template < int N >
void online_softmax(float* x, float* y, int N_, cudaEvent_t& start, cudaEvent_t& end){
	assert( N % 512 == 0 );
	
	cudaEventRecord(start);
	// 一个block 128线程 一个线程 4 * size / 512个数据, 长度必须是512的倍数
	// 算法没办法跨block
	kernel<N><<<1, 128>>>(x, y, N);
	cudaEventRecord(end);
	cudaEventSynchronize(end);
}

#define template_define_helper(NNN) \
template void online_softmax<NNN>(float* x, float* y, int N_, cudaEvent_t& start, cudaEvent_t& end);


template_define_helper(512)
template_define_helper(512*2)
template_define_helper(512*4)
template_define_helper(512*8)
template_define_helper(512*16)
template_define_helper(512*32)
template_define_helper(512*64)
template_define_helper(512*128)