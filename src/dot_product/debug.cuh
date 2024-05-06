#pragma once
#define checkCudaErrors(func)\
{\
	cudaError_t e = (func); \
	if(e != cudaSuccess) \
		printf ("%s %d CUDA: %s\n", __FILE__,  __LINE__, cudaGetErrorString(e)); \
}


#define EcudaMalloc(x1,x2) checkCudaErrors(cudaMalloc(x1,x2))
#define EcudaMemset(x1,x2,x3) checkCudaErrors(cudaMemset(x1,x2,x3))
#define EcudaMemcpy(x1,x2,x3,x4) checkCudaErrors(cudaMemcpy(x1,x2,x3,x4))
#define EcudaFree(x) checkCudaErrors(cudaFree(x))
