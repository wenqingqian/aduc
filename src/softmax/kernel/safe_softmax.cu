#include "float4.cuh"
#include <assert.h>
#include <cfloat>
#include "reduce.cuh"
#include "debug.cuh"
#include "atomicMaxFloat.cuh"

namespace {

__global__ 
void kernel(float* __restrict__ x, float* __restrict__ y, 
			float* __restrict__ sumexp, float* __restrict__ maxn, int N)
{
	// block内只有一个warp

	// pass 1
	int line = blockIdx.x * (128 / 4) + threadIdx.x;
	aduc::float4 data = reinterpret_cast<aduc::float4 *>(x)[line];
	
	float tmpmax = FLT_MIN;
	for( int i = 0; i < 4; i ++ ){
		tmpmax = fmaxf(tmpmax, data[i]);
	}
	tmpmax = warp_reduce_max(tmpmax);
	if( threadIdx.x == 0 ){
		aduc::atomicMaxFloat(maxn, tmpmax);
	}
	__threadfence();

	// pass 2
	float tmp = 0;
	for(int i = 0; i < 4; i++){
		data[i] = expf(data[i] - *maxn);
		tmp += data[i];
	}

	tmp = warp_reduce_sum(tmp);
	if( threadIdx.x == 0 ){
		atomicAdd(sumexp, tmp);
	}
	__threadfence();

	// pass 3
	aduc::float4 *out = reinterpret_cast<aduc::float4 *>(y);

	out[line] = data / *sumexp;
}

}


void safe_softmax(float* x, float* y, int N, cudaEvent_t& start, cudaEvent_t& end){
	assert( N % 128 == 0 );
	//一个block 一个warp, 一个线程四个数, 一个block处理128个数, 32个单位(float4)

	float *sumexp, *maxn;
	EcudaMalloc(&sumexp, sizeof(float));
	EcudaMemset(sumexp, 0, sizeof(float));
	EcudaMalloc(&maxn, sizeof(float));
	EcudaMemset(maxn, 0, sizeof(float));
	
	cudaEventRecord(start);
	kernel<<<N / 128, 32>>>(x, y, sumexp, maxn, N);
	cudaEventRecord(end);
	cudaEventSynchronize(end);
	
	float a1, a2;
	EcudaMemcpy(&a1, maxn, 4, cudaMemcpyDeviceToHost);
	EcudaMemcpy(&a2, sumexp, 4, cudaMemcpyDeviceToHost);

	EcudaFree(sumexp);
	EcudaFree(maxn);
}