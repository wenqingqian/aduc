#include <cuda_runtime.h>
#include "matrix.h"

extern void sigmoid(float*,float*,int);
extern void relu(float*,float*,int);


int main(){
	printf("sigmoid----------------------------------\n");
	{
		const int N = 8;
		matrix<float> ma(N,1);
		float* x;
		cudaMalloc(&x, sizeof(float) * N);
		cudaMemcpy(x, ma.unsafe_data(), sizeof(float) * N, cudaMemcpyHostToDevice);
		ma.show();
		sigmoid(x, x, N);
		cudaMemcpy(ma.unsafe_data(), x, sizeof(float) * N, cudaMemcpyDeviceToHost);
		cudaDeviceSynchronize();
		ma.show();
	}
	printf("relu----------------------------------\n");
	{
		const int N = 8;
		matrix<float> ma(N,1);
		float* x;
		cudaMalloc(&x, sizeof(float) * N);
		cudaMemcpy(x, ma.unsafe_data(), sizeof(float) * N, cudaMemcpyHostToDevice);
		ma.show();
		relu(x, x, N);
		cudaMemcpy(ma.unsafe_data(), x, sizeof(float) * N, cudaMemcpyDeviceToHost);
		cudaDeviceSynchronize();
		ma.show();
	}
}