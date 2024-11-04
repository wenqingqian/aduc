#include <cublas_v2.h>

#include <cuda_runtime.h>

#include "matrix.h"

#include "util.cuh"
int main(){
	int M = 3584, N = 3584, K = 3584;
	float alpha = 1.2, beta = 1.3;
	matrix<float> A(M,K), B(K,N), C(M,N);
	float * A_, * B_, * C_;
	
	cudaMalloc(&A_, M * K * sizeof(float));
	cudaMemcpy(A_, A.unsafe_data(), M * K * sizeof(float),
				cudaMemcpyHostToDevice);
	cudaMalloc(&B_, K * N * sizeof(float));
	cudaMemcpy(B_, B.unsafe_data(), K * N * sizeof(float),
				cudaMemcpyHostToDevice);
	cudaMalloc(&C_, M * N * sizeof(float));
	cudaMemcpy(C_, C.unsafe_data(), M * N * sizeof(float),
				cudaMemcpyHostToDevice);

	cudaDeviceSynchronize();

	extern_gemm(gemmHideGmemLatency);
	gemmHideGmemLatency(A_,B_,C_,alpha,beta,M,N,K);

	extern_gemm(r1_HideGmemLatency);
	r1_HideGmemLatency(A_,B_,C_,alpha,beta,M,N,K);
	cudaDeviceSynchronize();
}