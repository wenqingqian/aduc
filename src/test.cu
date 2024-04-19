#include <cublas_v2.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <eigen3/Eigen/Core>
#include <omp.h>
#include "util.cuh"

int _ConvertSMVer2Cores(int major, int minor) {
    // Defines for GPU Architecture types (using the SM version to determine the # of cores per SM)
    typedef struct {
        int SM; //  CUDA Cores/Streaming Multiprocessor
        int Cores;
    } sSMtoCores;

    sSMtoCores nGpuArchCoresPerSM[] = {
        { 0x10,  8 }, // Tesla Generation (SM 1.0) G80 class
        { 0x11,  8 }, // Tesla Generation (SM 1.1) G8x class
        { 0x12,  8 }, // Tesla Generation (SM 1.2) G9x class
        { 0x13,  8 }, // Tesla Generation (SM 1.3) GT200 class
        { 0x20, 32 }, // Fermi Generation (SM 2.0) GF100 class
        { 0x21, 48 }, // Fermi Generation (SM 2.1) GF10x class
        { 0x30, 192}, // Kepler Generation (SM 3.0) GK10x class
        { 0x32, 192}, // Kepler Generation (SM 3.2) GK10x class
        { 0x35, 192}, // Kepler Generation (SM 3.5) GK11x class
        { 0x37, 192}, // Kepler Generation (SM 3.7) GK21x class
        { 0x50, 128}, // Maxwell Generation (SM 5.0) GM10x class
        { 0x52, 128}, // Maxwell Generation (SM 5.2) GM20x class
        { 0x53, 128}, // Maxwell Generation (SM 5.3) GM20x class
        { 0x60, 64 }, // Pascal Generation (SM 6.0) GP100 class
        { 0x61, 128}, // Pascal Generation (SM 6.1) GP10x class
        { 0x62, 128}, // Pascal Generation (SM 6.2) GP10x class
        { 0x70, 64 }, // Volta Generation (SM 7.0) GV100 class
        { 0x72, 64 }, // Volta Generation (SM 7.2) GV10x class
        { 0x75, 64 }, // Turing Generation (SM 7.5) TU10x class
        { 0x80, 64 }, // Ampere Generation (SM 8.0) GA10x class
        { 0x86, 64 }, // Ampere Generation (SM 8.6) GA10x class
        { 0x90, 128}  // A100 Generation (SM 9.0) GA10x class
    };

    int index = 0;
    while (index < sizeof(nGpuArchCoresPerSM) / sizeof(nGpuArchCoresPerSM[0])) {
        if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor)) {
            return nGpuArchCoresPerSM[index].Cores;
        }
        index++;
    }
	printf(" undifined device \n");
    return -1;
}


// template < unsigned SM, unsigned CORE >
class kernel {
public:
	kernel(unsigned M, unsigned N, unsigned K, float alpha, float beta, int iterations=5):
		M_(M),N_(N),K_(K),alpha_(alpha),beta_(beta),HostResult_{M,N},DeviceResult_{M,N},iterations_(iterations)
	{
		Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> A{M, K}, B{K, N}, C{M, N};
		A.setRandom();
		B.setRandom();
		C.setRandom();

		cudaMalloc(&A_, M * K * sizeof(float));
		cudaMemcpy(A_, A.data(), M * K * sizeof(float),
					cudaMemcpyHostToDevice);
		cudaMalloc(&B_, K * N * sizeof(float));
		cudaMemcpy(B_, B.data(), K * N * sizeof(float),
					cudaMemcpyHostToDevice);
		cudaMalloc(&C_, M * N * sizeof(float));
		cudaMemcpy(C_, C.data(), M * N * sizeof(float),
					cudaMemcpyHostToDevice);

		HostResult_ = alpha * (A * B) + beta * C;

		cudaMalloc(&C_copy_, M * N * sizeof(float));
		cudaMemcpy(C_copy_, C.data(), M * N * sizeof(float),
					cudaMemcpyHostToDevice);
		
		cudaDeviceSynchronize();
		
		cuda_info();
	}

	void cuda_info(){
		int deviceCount;
		cudaGetDeviceCount(&deviceCount);
		if(deviceCount !=1){
			printf("multiple device, test with device 0\n");
		}
		
		int fp32CoresNum = 2560;
		int dev = 0;
		cudaDeviceProp deviceProp;
		cudaGetDeviceProperties(&deviceProp, dev);

		int fp32CoresNumPerSM = _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor);
		printf("Device %d: %s\n", dev, deviceProp.name);
		printf("  FP32 Cores Num: %d\n", fp32CoresNumPerSM);
		printf("  SM Num: %d\n", deviceProp.multiProcessorCount);
		fp32CoresNum = fp32CoresNumPerSM * deviceProp.multiProcessorCount;
	
		cudaSetDevice(dev);
		double boostFrequency = deviceProp.clockRate / 1e6;
		double peakPerformance = boostFrequency * fp32CoresNum * 2;
		printf("  FP32 peak throughput %.3f GFLOPS\n", peakPerformance);
		printf("\n");
	}

	template < class FUNC >
	void run(FUNC && func, const char* name){
		printf("kernel: %s\n",name);
		resetC();
		func(A_,B_,C_,alpha_,beta_,M_,N_,K_);
		checkValue();
		profile(std::forward<FUNC>(func));
	}

	template < class FUNC >
	void profile(FUNC && func){
		cudaEvent_t startEvent, stopEvent;
		cudaEventCreate(&startEvent);
		cudaEventCreate(&stopEvent);

		float millisecondsSum = 0;
		for (int i=0;i<iterations_;i++){
			resetC();
			cudaEventRecord(startEvent);
			func(A_,B_,C_,alpha_,beta_,M_,N_,K_);
			cudaEventRecord(stopEvent);
			cudaEventSynchronize(stopEvent);
			float milliseconds = 0;
			cudaEventElapsedTime(&milliseconds, startEvent, stopEvent);
			millisecondsSum += milliseconds;
		}
		millisecondsSum /= 1.0*iterations_;
		printf("  average GEMM time: %.3fms\n", millisecondsSum);
		double GFLOPS = 2 * 1e-9 * M_ * N_ * K_ / (millisecondsSum * 1e-3);
		printf("  average GFLOPS: %.3fGFLOPS\n", GFLOPS);

		cudaEventDestroy(stopEvent);
		cudaEventDestroy(startEvent);
		cudaDeviceSynchronize();
		printf("\n");
	}

	void resetC(){
		cudaMemcpy(C_, C_copy_, M_ * N_ * sizeof(float),
					cudaMemcpyDeviceToDevice);
	}

	void checkValue(){
		cudaMemcpy(DeviceResult_.data(), C_, M_ * N_ * sizeof(float),
					cudaMemcpyDeviceToHost);
		cudaDeviceSynchronize();
		Eigen::Array<float, Eigen::Dynamic, Eigen::Dynamic> diffArray =
			(HostResult_ - DeviceResult_).array().abs();
		printf("  Gemm Error: %f\n", diffArray.maxCoeff());
	}

private:
	float * A_, * B_, * C_, * C_copy_;
	unsigned M_, N_, K_;
	float alpha_, beta_;
	int iterations_;
	Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> HostResult_, DeviceResult_;
};

class gemmCuBlas {
	cublasHandle_t handle{nullptr};
public:
	gemmCuBlas() { cublasCreate(&handle); }
	~gemmCuBlas() { cublasDestroy(handle); }

	void operator()(const float *A, const float *B, float *C, float &alpha,
					float &beta, int M, int N, int K) const {
		int lda = N, ldb = K, ldc = N;
		cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B, lda, A,
					ldb, &beta, C, ldc);
	}
};

extern_gemm(gemmNaive)
extern_gemm(gemmTile)

int main(){
	omp_set_num_threads(omp_get_num_procs());
	printf("start proc: %d\n\n", omp_get_num_procs());

	unsigned M = 2048, N = 2048, K = 1024;
	float alpha = 1.5, beta = 1.6;
	kernel k(M,N,K,alpha,beta,10);

	k.run(gemmCuBlas{}, "gemmCuBlas");
	k.run(gemmNaive, "gemmNaive");
	k.run(gemmTile, "gemmTile");
}