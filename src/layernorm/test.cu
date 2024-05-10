#include <cuda_runtime.h>
#include "matrix.h"

extern void layernorm1d(float*, float*, float*, float*, float, int, int);


int main(){
	const int batch = 2;
	const int N = 8;
	const float eps = 0.00001;
	matrix<float> mx(batch, N);
	matrix<float> mw(1, N);
	matrix<float> mb(1, N);
	// for(int i = 0; i < N; i++){
	// 	mw[i] = 1;
	// 	mb[i] = 0;
	// }
	
	// compare matrix
	matrix<float> c = mx;
	for(int i=0;i<batch;i++){
		float mean = 0;
		float var = 0;
		for(int j=0;j<N;j++){
			mean += c[i*N+j];
		}
		mean /= N;
		for(int j=0;j<N;j++){
			c[i*N+j] -= mean;
			var += (c[i*N+j] * c[i*N+j]);
		}
		var = sqrt(var / N + eps);
		for(int j=0;j<N;j++){
			c[i*N+j] = mw[j] * c[i*N+j] / var + mb[j];
		}
	}
	c.show("compare layernorm");
	

	float* x, *weight, *bias;
	cudaMalloc(&x, sizeof(float) * N * batch);
	cudaMemcpy(x, mx.unsafe_data(), sizeof(float) * N * batch, cudaMemcpyHostToDevice);
	cudaMalloc(&weight, sizeof(float) * N);
	cudaMemcpy(weight, mw.unsafe_data(), sizeof(float) * N, cudaMemcpyHostToDevice);
	cudaMalloc(&bias, sizeof(float) * N);
	cudaMemcpy(bias, mb.unsafe_data(), sizeof(float) * N, cudaMemcpyHostToDevice);
	mx.show("origin layernorm mx");

	layernorm1d(x, x, weight, bias, eps, batch, N);
	cudaMemcpy(mx.unsafe_data(), x, sizeof(float) * N * batch, cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	mx.show("result layernorm mx");

	printf("error: %f\n", matrix<float>::diff(mx, c));

}