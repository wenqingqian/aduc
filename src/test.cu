#include <cublas_v2.h>
#include <stdio.h>

void cuda_info(){
	int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    for (int dev = 0; dev < deviceCount; ++dev) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        int fp32CoresNum = 128*20;
        printf("Device %d: %s\n", dev, deviceProp.name);
        printf("  FP32 Cores Num: %d\n", fp32CoresNum);
		printf("  SM Num: %d\n", deviceProp.multiProcessorCount);

        printf("\n");
    }

	int gpu_rank = 0;
	cudaDeviceProp deviceProp{};
	cudaGetDeviceProperties(&deviceProp, gpu_rank);
	cudaSetDevice(gpu_rank);
	double boostFrequency = deviceProp.clockRate / 1e6;
	int fp32CoresNum = 2560;
	double peakPerformance = boostFrequency * fp32CoresNum * 2;
	printf("FP32 peak throughput %.3f GFLOPS\n", peakPerformance);
}

__global__ void gemmKernel(const float * A,
	const float * B, float * C,
	float alpha, float beta, unsigned M, unsigned N,
	unsigned K) {
	unsigned int m = threadIdx.x + blockDim.x * blockIdx.x;
	unsigned int n = threadIdx.y + blockDim.y * blockIdx.y;
	if (m >= M || n >= N)
		return;
	float c = 0;
	for (unsigned k = 0; k < K; ++k) {
		c += A[m * K + k] * B[k * N + n];
	}
	c = c * alpha;
	float result = c;
	if (beta != 0) {
		result = result + C[m * N + n] * beta;
	}
	C[m * N + n] = result;
}

void gemmNaive(const float *A, const float *B, float *C,
	float alpha, float beta, unsigned M,
	unsigned N, unsigned K) {
	dim3 block(16, 16);
	dim3 grid((M - 1) / block.x + 1, (N - 1) / block.y + 1);

	gemmKernel<<<grid, block>>>(A, B, C, alpha, beta, M, N, K);
}

void cublasGemm(const float *A, const float *B, float *C, float alf, float bet, int M, int N, int K) {
	int lda = N, ldb = K, ldc = N;
	const float *alpha = &alf;
	const float *beta = &bet;
	cublasHandle_t handle;
	cublasCreate(&handle);

	cudaEvent_t startEvent, stopEvent;
	cudaEventCreate(&startEvent);
	cudaEventCreate(&stopEvent);
	cudaEventRecord(startEvent);

	cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, alpha, B, lda, A, ldb, beta, C, ldc);

	cudaEventRecord(stopEvent);
	cudaEventSynchronize(stopEvent);

	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, startEvent, stopEvent);
	printf("cublas Average Time: %.3f ms\n", milliseconds);
	
	cudaEventDestroy(stopEvent);
	cudaEventDestroy(startEvent);

	cublasDestroy(handle);
	cudaDeviceSynchronize();

	double GFLOPS = 2 * 1e-9 * M * N * K / (milliseconds * 1e-3);
	printf("Average Throughput: %.3f GFLOPS\n", GFLOPS);
}

void cuda_main(float* A, float* B, float* C, unsigned M, unsigned N, unsigned K, 
				float alpha, float beta, float* callback) {
	cuda_info();


	float *deviceAPrt, *deviceBPtr, *deviceCPtr;

	cudaMalloc(&deviceAPrt, M * K * sizeof(float));
	cudaMemcpy(deviceAPrt, A, M * K * sizeof(float),
				cudaMemcpyHostToDevice);
	cudaMalloc(&deviceBPtr, K * N * sizeof(float));
	cudaMemcpy(deviceBPtr, B, K * N * sizeof(float),
				cudaMemcpyHostToDevice);
	cudaMalloc(&deviceCPtr, M * N * sizeof(float));
	cudaMemcpy(deviceCPtr, C, M * N * sizeof(float),
				cudaMemcpyHostToDevice);

	cublasGemm(deviceAPrt, deviceBPtr, deviceCPtr, alpha, beta, M, N, K);


	cudaMemcpy(callback, deviceCPtr, M * N * sizeof(float),
			cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
}
