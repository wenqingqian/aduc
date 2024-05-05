#include <eigen3/Eigen/Core>
#include <omp.h>
#include <cuda_runtime.h>
#include <iostream>
#include "sgemv.cuh"
#include "debug.cuh"
#include "matrix.h"
#include <string>

class sgemv {
public:
	sgemv(int m, int n): M(m), N(n), y_h{M, 1, false, false}, y_r{M, 1, true, false}
	{
		// printf("debug sgemv %d-%d\n",M,N);
		matrix<float> A_h(M, N);
		matrix<float> x_h(N, 1);
		
		EcudaMalloc(&A_d, M * N * sizeof(float));
		EcudaMemcpy(A_d, A_h.unsafe_data(), M * N * sizeof(float), cudaMemcpyHostToDevice);
		EcudaMalloc(&x_d, N * 1 * sizeof(float));
		EcudaMemcpy(x_d, x_h.unsafe_data(), N * 1 * sizeof(float), cudaMemcpyHostToDevice);
		EcudaMalloc(&y_d, M * 1 * sizeof(float));
		EcudaMemset(y_d, 0, M * 1 * sizeof(float));

		y_h = A_h * x_h;

		cudaDeviceSynchronize();
		// printf("debug %d-%d-%d-%d\n",y_h.M,y_h.N,y_r.M,y_r.N);

	}
	~sgemv(){
		EcudaFree(A_d);
		EcudaFree(x_d);
		EcudaFree(y_d);
	}

	template < class FUNC >
	void run( FUNC&& func, const std::string& str) {
		func(A_d, x_d, y_d, M, N);
		EcudaMemcpy(y_r.unsafe_data(), y_d, M * 1 * sizeof(float), cudaMemcpyDeviceToHost);
		cudaDeviceSynchronize();

		// printf("debug %d-%d-%d-%d\n",y_h.M,y_h.N,y_r.M,y_r.N);
		// y_h.show();
		// y_r.show();

		std::cout<<str<<" Error: "<<matrix<float>::diff(y_r, y_h)<<std::endl;
	}

private:
	// 类内成员初始化顺序与构造函数列表中顺序无关, 与成员在类中声明的顺序相关
	// M, N 要在 matrix 前面
	const int M;
	const int N;

	matrix<float> y_h;
	matrix<float> y_r;

	float* A_d, *x_d, *y_d;
};


template < int N >
void test(int a){
	if(a!=0 && a != N) return;
	const int M  = 128;
	sgemv s(M, N);

	// 一定要用if constexpr, 不然递归编译下去会把下面三个情况都编译
	// 当用 N < 4 编译第三条 sgemv_gt32<N> 时, 会在sgemv_gt32 的kernel中继续编译
	// constexpr int row_per_warp = 32 / thread_per_row; 此处thread_per_row 为 0
	// 触发编译期计算时用条件编译控制范围
	if constexpr (N < 32)
		s.run(sgemv_lt32<N>, "sgemv_"+std::to_string(N));
	else if constexpr (N == 32)
		s.run(sgemv_eq32, "sgemv_32");
	else
		s.run(sgemv_gt32<N>, "sgemv_"+std::to_string(N));
}

template < int begin, int end >
void staticTest(int N){
	test<begin>(N);
	if constexpr ( begin < 32 ){
		staticTest<begin+1, end>(N);
	} else if constexpr ( begin != end ){
		staticTest<begin+4, end>(N);
	}
}
int main(int argc, char* argv[]){
	omp_set_num_threads(omp_get_num_procs());

	if( argc > 1 ){
		int N = std::stoi(argv[1]);
		// 一次性生成所有案例, 但是代码存在奇怪的bug, 只能一个一个测试:(
		// bug 已解决, 类内成员初始化顺序错误
		staticTest<1,128>(N);
	}else{
		staticTest<1,128>(0);
	}
	
}


// class sgemv {
// public:
// 	sgemv(int m, int n): M(m), N(n), y_h{M, 1}, y_r{M, 1}
// 	{
// 		Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> A_h{M, N};
// 		Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> x_h{N, 1};

// 		A_h.setRandom();
// 		x_h.setRandom();
		
// 		EcudaMalloc(&A_d, M * N * sizeof(float));
// 		EcudaMemcpy(A_d, A_h.data(), M * N * sizeof(float), cudaMemcpyHostToDevice);
// 		EcudaMalloc(&x_d, N * 1 * sizeof(float));
// 		EcudaMemcpy(x_d, x_h.data(), N * 1 * sizeof(float), cudaMemcpyHostToDevice);
// 		EcudaMalloc(&y_d, M * 1 * sizeof(float));
// 		EcudaMemset(y_d, 0, M * 1 * sizeof(float));

// 		y_h = A_h * x_h;

// 		cudaDeviceSynchronize();
// 	}
// 	~sgemv(){
// 		EcudaFree(A_d);
// 		EcudaFree(x_d);
// 		EcudaFree(y_d);
// 	}

// 	template < class FUNC >
// 	void run( FUNC&& func, const char* str) {
// 		func(A_d, x_d, y_d, M, N);
// 		EcudaMemcpy(y_r.data(), y_d, M * 1 * sizeof(float), cudaMemcpyDeviceToHost);
// 		cudaDeviceSynchronize();
// 		Eigen::Array<float, Eigen::Dynamic, Eigen::Dynamic> diffArray =
// 			(y_h - y_r).array().abs();
// 		printf("%s Error: %f\n", str, diffArray.maxCoeff());
// 	}

// private:
// 	Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> y_h;
// 	Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> y_r;
// 	const int M;
// 	const int N;
// 	float* A_d, *x_d, *y_d;
// };