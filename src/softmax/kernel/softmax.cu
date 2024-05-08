#include "float4.cuh"
#include <assert.h>
#include "reduce.cuh"
#include "debug.cuh"
namespace {

__global__ 
void kernel(float* __restrict__ x, float* __restrict__ y, float* __restrict__ sumexp, int N){
	int line = blockIdx.x * (128 / 4) + threadIdx.x;
	aduc::float4 data = reinterpret_cast<aduc::float4 *>(x)[line];
	
	float tmp = 0;
	for(int i = 0; i < 4; i++){
		data[i] = expf(data[i]);
		tmp += data[i];
	}

	tmp = warp_reduce_sum(tmp);
	if( threadIdx.x == 0 ){
		atomicAdd(sumexp, tmp);
	}
	__threadfence();

	aduc::float4 *out = reinterpret_cast<aduc::float4 *>(y);

	out[line] = data / *sumexp;
}

}


void softmax(float* x, float* y, int N, cudaEvent_t& start, cudaEvent_t& end){
	assert( N % 128 == 0 );
	//一个block 一个warp, 一个线程四个数, 一个block处理128个数, 32个单位(float4)

	float *sumexp;
	EcudaMalloc(&sumexp, sizeof(float));
	EcudaMemset(sumexp, 0, sizeof(float));

	cudaEventRecord(start);
	kernel<<<N / 128, 32>>>(x, y, sumexp, N);
	cudaEventRecord(end);
	cudaEventSynchronize(end);
	EcudaFree(sumexp);
}