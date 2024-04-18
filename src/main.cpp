#include <omp.h>
#include <eigen3/Eigen/Core>

using namespace Eigen;


extern void cuda_main(float* A, float* B, float* C, unsigned M, unsigned N, unsigned K, 
				float alpha, float beta, float* callback);

int main(){
	unsigned M = 2048, N = 2048, K = 1024;
	float alpha = 1., beta = 1.;

	Matrix<float, Dynamic, Dynamic, RowMajor> A{M, K}, B{K, N}, C{M, N};
	A.setRandom();
	B.setRandom();
	C.setRandom();

	Matrix<float, Dynamic, Dynamic, RowMajor> hostResult{M, N}, deviceResult{M, N};

	cuda_main(A.data(), B.data(), C.data(), M, N, K, alpha, beta, deviceResult.data());

	omp_set_num_threads(omp_get_num_procs());
	clock_t begin, end;
	begin = clock();
	hostResult = alpha * (A * B) + beta * C;
	end = clock();
	printf("diff Average Time: %.3f ms\n", double(end - begin) / CLOCKS_PER_SEC * 1e3);


	Eigen::Array<float, Eigen::Dynamic, Eigen::Dynamic> diffArray =
		(hostResult - deviceResult).array().abs();
	printf("Max Error: %f\n", diffArray.maxCoeff());
	
}