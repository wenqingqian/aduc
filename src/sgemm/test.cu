#include <cublas_v2.h>
#include <stdio.h>
#include <cuda_runtime.h>
// #include <eigen3/Eigen/Core>
#include "matrix.h"
#include <omp.h>
#include "util.cuh"
#include <fstream>

int _ConvertSMVer2Cores(int major, int minor) {
    // Defines for GPU Architecture types (using the SM version to determine the # of cores per SM)
    typedef struct {
        int SM; //  CUDA Cores/Streaming Multiprocessor
        int Cores;
    } sSMtoCores;

    sSMtoCores nGpuArchCoresPerSM[] = {
        { 0x10,  8 }, // Tesla Generation (SM 1.0) G80 class
        { 0x11,  8 }, // Tesla Generation (SM 1.1) G8x class·
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
        { 0x86, 128 }, // Ampere Generation (SM 8.6) GA10x class
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
		#ifdef STORE_RESULT
		ofs.open("output.txt", std::ios::app);
		// ofs << "name,M,N,K,alpha,beta,error,time,gflops,compare_to_cublas,compare_to_peak\n";
		#endif
		printf("KERNEL : ------------\n");
		printf("M: %d\n", M_);
		printf("N: %d\n", N_);
		printf("K: %d\n", K_);
		printf("alpha: %f\n", alpha);
		printf("beta: %f\n", beta);

		cudaEventCreate(&startEvent_);
		cudaEventCreate(&stopEvent_);
		
		// Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> A{M, K}, B{K, N}, C{M, N};
		// A.setRandom();
		// B.setRandom();
		// C.setRandom();
		matrix<float> A(M,K), B(K,N), C(M,N);

		cudaMalloc(&A_, M * K * sizeof(float));
		cudaMemcpy(A_, A.unsafe_data(), M * K * sizeof(float),
					cudaMemcpyHostToDevice);
		cudaMalloc(&B_, K * N * sizeof(float));
		cudaMemcpy(B_, B.unsafe_data(), K * N * sizeof(float),
					cudaMemcpyHostToDevice);
		cudaMalloc(&C_, M * N * sizeof(float));
		cudaMemcpy(C_, C.unsafe_data(), M * N * sizeof(float),
					cudaMemcpyHostToDevice);

		// for(int i = 0; i < M; i ++){
		// 	for(int j = 0; j < N; j ++){
		// 		float tmp = 0;
		// 		for(int a = 0; a<K; a++){
		// 			tmp += A[i*K+a] * B[a*N+j];
		// 		}
		// 		HostResult_[i*N+j] = tmp + C[i*N+j];
		// 	}
		// }
		#ifndef DISABLE_CPU
		HostResult_ = alpha * A * B + beta * C;
		#endif

		cudaMalloc(&C_copy_, M * N * sizeof(float));
		cudaMemcpy(C_copy_, C.unsafe_data(), M * N * sizeof(float),
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
		printf("  ShareMemSize: %d\n", deviceProp.sharedMemPerBlock);
		fp32CoresNum = fp32CoresNumPerSM * deviceProp.multiProcessorCount;
	
		cudaSetDevice(dev);
		double boostFrequency = deviceProp.clockRate / 1e6;
		double peakPerformance = boostFrequency * fp32CoresNum * 2;
		peakPerformance_ = peakPerformance;
		printf("  FP32 peak throughput %.3f GFLOPS\n", peakPerformance);
		printf("\n");
	}

	template < class FUNC >
	void run(FUNC && func, const char* name){
		printf("kernel: %s\n",name);
		#ifdef STORE_RESULT
		ofs << name << ',' << M_ << ',' << N_ << ',' << K_ << ',' << alpha_ << ',' << beta_ << ',';
		#endif

		resetC();
		func(A_,B_,C_,alpha_,beta_,M_,N_,K_);
		checkValue();
		for(int i=0;i<50;i++){
			func(A_,B_,C_,alpha_,beta_,M_,N_,K_);
		}
		profile(std::forward<FUNC>(func));
	}

	template < class FUNC >
	void profile(FUNC && func){
		
		float millisecondsSum = 0;
		for (int i=0;i<iterations_;i++){
			resetC();
			cudaEventRecord(startEvent_);
			func(A_,B_,C_,alpha_,beta_,M_,N_,K_);
			cudaEventRecord(stopEvent_);
			cudaEventSynchronize(stopEvent_);
			float milliseconds = 0;
			cudaEventElapsedTime(&milliseconds, startEvent_, stopEvent_);
			millisecondsSum += milliseconds;
		}
		millisecondsSum /= 1.0*iterations_;
		printf("  average GEMM time: %.3fms\n", millisecondsSum);
		double GFLOPS = 2 * 1e-9 * M_ * N_ * K_ / (millisecondsSum * 1e-3);
		printf("  average GFLOPS: %.3fGFLOPS\n", GFLOPS);
		if(cublasPerformance_ == 0) cublasPerformance_ = GFLOPS;

		double p_cublas = GFLOPS * 100.0 / cublasPerformance_;
		double p_peak   = GFLOPS * 100.0 / peakPerformance_;
		printf("  compare with cublas: %.3f%\n", p_cublas);
		printf("  compare with peak: %.3f%\n", p_peak);
		printf("\n");

		#ifdef STORE_RESULT
		char buffer[64];
		std::snprintf(buffer, sizeof(buffer), "%.2f,%.2f,%.2f,%.2f", millisecondsSum, GFLOPS, p_cublas, p_peak);
		ofs << buffer << std::endl;
		#endif
	}

	void resetC(){
		cudaMemcpy(C_, C_copy_, M_ * N_ * sizeof(float),
					cudaMemcpyDeviceToDevice);
	}

	void checkValue(){
		cudaMemcpy(DeviceResult_.unsafe_data(), C_, M_ * N_ * sizeof(float),
					cudaMemcpyDeviceToHost);
		cudaDeviceSynchronize();

		// Eigen::Array<float, Eigen::Dynamic, Eigen::Dynamic> diffArray =
		// 	(HostResult_ - DeviceResult_).array().abs();
		// printf("  Gemm Error: %f\n", diffArray.maxCoeff());
		float diff = matrix<float>::diff(DeviceResult_, HostResult_);
		printf("  Gemm Error: %f%\n", diff);



		#ifdef DISABLE_CPU
		if(cublasPerformance_ == 0){
			HostResult_ = DeviceResult_;
		}else{
			if(diff > 1){
				printf("diff error\n");
				exit(0);
			}
		}
		#else
		if(diff > 1){
			printf("diff error\n");
			exit(0);
		}
		#endif

		#ifdef STORE_RESULT
		char buffer[16];
		std::snprintf(buffer, sizeof(buffer), "%.2f,", diff);
		ofs << buffer;
		#endif
	}

private:
	cudaEvent_t startEvent_, stopEvent_;

	float * A_, * B_, * C_, * C_copy_;
	unsigned M_, N_, K_;
	float alpha_, beta_;
	int iterations_;
	// Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> HostResult_, DeviceResult_;
	matrix<float> HostResult_, DeviceResult_;
	std::ofstream ofs;
	float peakPerformance_;
	float cublasPerformance_ = 0;
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



// extern_gemm(gemmNaive)
// extern_gemm(gemmTile)
// extern_gemm(gemmShareMem)
// extern_gemm(r1_ShareMem)
extern_gemm(gemmColMajorSMA)
// extern_gemm(r1_ColMajorSMA)
extern_gemm(r1_HideSmemLatency)
extern_gemm(gemmHideSmemLatency)
extern_gemm(gemmHideGmemLatency)
extern_gemm(r1_HideGmemLatency)

int main(int argc, char** argv){
	// omp_set_num_threads(omp_get_num_procs());
	// printf("start proc: %d\n\n", omp_get_num_procs());

	assert(argc == 4);
	unsigned M = std::stoi(argv[1]);
	unsigned N = std::stoi(argv[2]);
	unsigned K = std::stoi(argv[3]);
	float alpha = 1.2;
	float beta = 1.3;
	
	// unsigned M = 20*128, N = 20*128, K = 1024;
	// float alpha = 1.5, beta = 1.6;
	kernel k(M,N,K,alpha,beta,100);

	k.run(gemmCuBlas{}, "gemmCuBlas");
	// k.run(gemmNaive, "gemmNaive");
	// k.run(gemmTile, "gemmTile");
	// k.run(r1_ShareMem, "r1_ShareMem");
	// k.run(gemmShareMem, "gemmShareMem");
	k.run(r1_HideSmemLatency, "r1_HideSmemLatency");
	k.run(gemmHideSmemLatency, "gemmHideSmemLatency");
	k.run(gemmColMajorSMA, "gemmColMajorSMA");
	// k.run(r1_ColMajorSMA, "r1_ColMajorSMA");
	k.run(r1_HideGmemLatency, "r1_HideGmemLatency");
	k.run(gemmHideGmemLatency, "gemmHideGmemLatency");
	
}