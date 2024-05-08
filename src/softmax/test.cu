#include <cuda_runtime.h>
#include <iostream>
#include "matrix.h"
#include "debug.cuh"
#include <omp.h>

enum { Safe, Unsafe };
template < int type >
class softmaxtest{
public:
	softmaxtest(int n): N(n), y_h(N, 1, false, false), y_r(N, 1, true, false)
	{
		cudaEventCreate(&startEvent_);
		cudaEventCreate(&stopEvent_);

		matrix<float> x_h(N, 1);
		
		EcudaMalloc(&x_d, N * 1 * sizeof(float));
		EcudaMemcpy(x_d, x_h.unsafe_data(), N * 1 * sizeof(float), cudaMemcpyHostToDevice);
		EcudaMalloc(&y_d, N * 1 * sizeof(float));
		EcudaMemset(y_d, 0, N * 1 * sizeof(float));

		if constexpr (type == Unsafe)
			y_h = std::move(cpu_softmax(x_h));
		else
			y_h = std::move(cpu_safe_softmax(x_h));

		cudaDeviceSynchronize();
	}
	~softmaxtest(){
		EcudaFree(x_d);
		EcudaFree(y_d);
	}

	template < class FUNC >
	void run( FUNC&& func, const std::string& str, int iterations = 1) {
		func(x_d, y_d, N, startEvent_, stopEvent_);
		EcudaMemcpy(y_r.unsafe_data(), y_d, N * 1 * sizeof(float), cudaMemcpyDeviceToHost);
		cudaDeviceSynchronize();

			// y_h.show();
			// y_r.show();

		std::cout<<str<<"\n  Error: "<<matrix<float>::diff(y_r, y_h)<<std::endl;

		if(iterations > 1) profile(std::forward<FUNC>(func), iterations);
	}

	template < class FUNC >
	void profile( FUNC&& func, int iterations){
		float millisecondsSum = 0;
		for (int i=0;i<iterations;i++){
			func(x_d, y_d, N, startEvent_, stopEvent_);
			float milliseconds = 0;
			cudaEventElapsedTime(&milliseconds, startEvent_, stopEvent_);
			millisecondsSum += milliseconds;
		}
		millisecondsSum /= 1.0*iterations;
		printf("  average softmax time: %fms\n", millisecondsSum);
	}

	template< class T >
	matrix<T> cpu_softmax(matrix<T> ma){
		T sum = T(0);
		for( int i = 0; i < ma.M*ma.N; i ++ ){
			ma[i] = expf(ma[i]);
			sum += ma[i];
		}

		for( int i = 0; i < ma.M*ma.N; i ++ ){
			ma[i] /= sum;
		}

		// 用move转, 现在似乎不用move都能自动move, RVO机制
		return ma;
	}
	
	template< class T >
	matrix<T> cpu_safe_softmax(matrix<T> ma){
		// 2 pass online softmax
		T maxn = T(0);
		T sum = T(0);
		for( int i = 0; i < ma.M*ma.N; i ++ ){
			T prevmaxn = maxn;
			maxn = max(maxn, ma[i]);
			sum  = sum * expf(prevmaxn - maxn) + expf(ma[i] - maxn);
		}

		for( int i = 0; i < ma.M*ma.N; i ++ ){
			ma[i] = expf(ma[i] - maxn) / sum;
		}
		// printf("maxn: %f, sum: %f\n", maxn, sum);
		return ma;
	}

private:
	const int N;

	matrix<float> y_h;
	matrix<float> y_r;

	float *x_d, *y_d;

	cudaEvent_t startEvent_, stopEvent_;
};

extern void softmax(float*,float*,int,cudaEvent_t&,cudaEvent_t&);
extern void safe_softmax(float*,float*,int,cudaEvent_t&,cudaEvent_t&);

template<int N> void online_softmax(float*, float*, int, cudaEvent_t&, cudaEvent_t&);


#define template_test(NNN)\
	{\
		constexpr int N = NNN;\
		printf("N = %d\n", N);\
		{\
			softmaxtest<Unsafe> smt(N);\
			smt.run(softmax, "unsafe_softmax", 1000);\
		}\
		{\
			softmaxtest<Safe> smt(N);\
			smt.run(safe_softmax, "3 pass safe_softmax", 1000);\
		}\
		{\
			softmaxtest<Safe> smt(N);\
			smt.run(online_softmax<N>, "2 pass online_softmax", 1000);\
		}\
	}\

int main(){
	omp_set_num_threads(omp_get_num_procs());

	template_test(512)
	template_test(512*2)
	template_test(512*4)
	template_test(512*8)
	template_test(512*16)
	template_test(512*32)
	template_test(512*64)
	template_test(512*128)

}