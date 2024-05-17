#include <cuda_runtime.h>
#include <assert.h>
#include "cpu_flashattn.h"
#include "matrix.h"
#include "debug.cuh"
#include <vector>
#include <cfloat>
#include <chrono>

namespace {
	// Modified from: https://github.com/tspeterkim/flash-attention-minimal/blob/main/flash.cu
	__global__ // flash_attention_forward_v1
	void kernel(
		float* __restrict__ Q, // N x d
		float* __restrict__ K, // N x d
		float* __restrict__ V, // N x d
		int N, // sequence length
		int d, // hidden dimension W
		const int Tr, // see paper
		const int Tc,
		const int Br,
		const int Bc,
		const float softmax_scale,
		float* __restrict__ l, // N, store the prev sequence summation
		float* __restrict__ m, // N, store the prev sequence max for online softmax
		float* __restrict__ Out // N x d
	){
		extern __shared__ float sram[];
		float* Qi = sram;
		float* Kj = Qi + (Br * d);
		float* Vi = Kj + (Bc * d);
		float* S  = Vi + (Bc * d); 

		int tid = threadIdx.x;
		
		if(tid >= Bc) return;

		// outer loop K, V
		for ( int j = 0; j < Tc; j ++ ){	

			// load Tile K, V to shared memory
			for ( int x = 0; x < d; x ++ ){
				Kj[tid * d + x] = K[j * Bc * d + tid * d + x];
				Vi[tid * d + x] = V[j * Bc * d + tid * d + x];
			}
			__syncthreads();

			// inner loop Q
			for ( int i = 0; i < Tr; i ++ ){
				for ( int x = 0; x < d; x ++ ){
					Qi[tid * d + x] = Q[i * Br * d + tid * d + x];
				}

				// xi = Q[k,:]K^T[:,i]
				// mi = max(mi-1, xi)
				float max_tmp = 0;
				for ( int x = 0; x < Bc; x ++ ){

					float sum = 0;
					for ( int y = 0; y < d; y ++ ){
						sum += Qi[tid * d + y] * Kj[x * d + y];
					}
					sum *= softmax_scale;
					
					max_tmp = fmaxf(max_tmp, sum);

					S[tid * Bc + x] = sum;
				}

				// printf("tx:%d : %f\n", tid, max_tmp);
				// di = di-1 + e^{xi-mi}
				float sum_tmp = 0;
				for ( int x = 0; x < Bc; x ++ ){
					S[tid * Bc + x] = expf(S[tid * Bc + x] - max_tmp);
					sum_tmp += S[tid * Bc + x];
				}

				// load prev sequence max and sum, l m change with outer loop (same index)
				float max_prev = m[i * Br + tid];
				float sum_prev = l[i * Br + tid];

				float max_new = max(max_tmp, max_prev);
				float sum_new = sum_prev * expf(max_prev - max_new) + sum_tmp * expf(max_tmp - max_new);

				m[i * Br + tid] = max_new;
				l[i * Br + tid] = sum_new;

				// oi
				for ( int x = 0; x < d; x ++ ){
					float sum = 0;
					for ( int y = 0; y < Bc; y ++ ){
						sum += S[tid * Bc + y] * Vi[y * d + x];
					}
					Out[i * Br * d + tid * d + x] = 
						( sum_prev * expf(max_prev - max_new) * Out[i * Br * d + tid * d + x] + expf(max_tmp - max_new) * sum ) 
						/ sum_new;
				}
			}
			__syncthreads(); // next outer loop will change the Kj Vi, ensuring that the use of Kj, Vi has ended in this loop
		}
	}
	
	__global__ // flash_attention_forward_v2
	void kernelV2(
		float* __restrict__ Q, // N x d
		float* __restrict__ K, // N x d
		float* __restrict__ V, // N x d
		int N, // sequence length
		int d, // hidden dimension W
		const int Tr, // see paper
		const int Tc,
		const int Br,
		const int Bc,
		const float softmax_scale,
		float* __restrict__ Out // N x d
	){
		extern __shared__ float sram[];
		float* Qi = sram;
		float* Kj = Qi + (Br * d);
		float* Vi = Kj + (Bc * d);
		float* S  = Vi + (Bc * d);
		float* Oi = S  + (Bc * Br);

		int tid = threadIdx.x;
		
		if(tid >= Bc) return;

		// outer loop Q
		for ( int i = 0; i < Tr; i ++ ){	

			// load Tile Q to shared memory
			for ( int x = 0; x < d; x ++ ){
				Qi[tid * d + x] = Q[i * Br * d + tid * d + x];
			}

			float max_row = 0;
			float sum_row  = 0;
			// inner loop K, V
			for ( int j = 0; j < Tc; j ++ ){
				for ( int x = 0; x < d; x ++ ){
					Kj[tid * d + x] = K[j * Bc * d + tid * d + x];
					Vi[tid * d + x] = V[j * Bc * d + tid * d + x];
				}
				__syncthreads();

				// xi = Q[k,:]K^T[:,i]
				// mi = max(mi-1, xi)
				float max_tmp = 0;
				for ( int x = 0; x < Bc; x ++ ){

					float sum = 0;
					for ( int y = 0; y < d; y ++ ){
						sum += Qi[tid * d + y] * Kj[x * d + y];
					}
					sum *= softmax_scale;
					
					max_tmp = fmaxf(max_tmp, sum);

					S[tid * Bc + x] = sum;
				}

				// printf("tx:%d : %f\n", tid, max_tmp);
				// di = di-1 + e^{xi-mi}
				float sum_tmp = 0;
				for ( int x = 0; x < Bc; x ++ ){
					S[tid * Bc + x] = expf(S[tid * Bc + x] - max_tmp);
					sum_tmp += S[tid * Bc + x];
				}


				float max_new = max(max_tmp, max_row);
				float sum_new = sum_row * expf(max_row - max_new) + sum_tmp * expf(max_tmp - max_new);


				// oi
				for ( int x = 0; x < d; x ++ ){
					float sum = 0;
					for ( int y = 0; y < Bc; y ++ ){
						sum += S[tid * Bc + y] * Vi[y * d + x]; // bank conflict?
					}

					if( j == 0 ){
						Oi[tid * d + x] = expf(max_tmp - max_new) * sum / sum_new; 
					} else {
						Oi[tid * d + x] = 
						( sum_row * expf(max_row - max_new) * Oi[tid * d + x] + expf(max_tmp - max_new) * sum ) 
						/ sum_new; 
					}
					
				}
				max_row = max_new;
				sum_row = sum_new;
			}

			for( int j = 0; j < d; j ++ ){
				Out[i * Br * d + tid * d + j] = Oi[tid * d + j];
			}

			__syncthreads(); // ensuring that the use of Kj, Vi has ended in this loop
		}
	}
}


std::vector<float> flash_attention_forward(
	const int N, const int d
){
	//For simplicity, only the calculation of single-head attention is considered, so as to avoid complex offsets of QKVlm in the code.
	//  and I would make certain assumptions about the matrix shape

	//Actually, for multi-head attention, we can apply Batch x head_nums block, each block perform a single-head attention with thread_num Bc
	//dim3 grid = (Batch, head_nums);
	
	// shared memory will be divided into 4 parts for Q[x:Br+x,:], K[:,x:Bc+x], V[x:Br+x,:] and the temporary matrix S = Q[x:Br+x]K[:,x:Bc+x]^T
	// Q: Br x d
	// K: Bc x d
	// V: Bc x d
	// S: Br x Bc
	const int Bc = 32;

	const int Br = 32;

	assert(N % Bc == 0 && N % Br == 0);
	const int Tc = N / Bc;
	
	const int Tr = N / Br;

	const float softmax_scale = 1.0 / sqrt(d);

	const int sram_size = (2 * (Bc * d) + Br * (Bc + d)) * sizeof(float);

	matrix<float> Qm(N, d), Km(N, d), Vm(N, d), lm(N, 1), mm(N, 1), Outm(N, d), Out_compare1(N, d), Out_compare2(N, d), OutmV2(N, d);

	for(int i = 0; i < N; i ++){
		lm[i] = 0;
		mm[i] = std::numeric_limits<float>::min();
	}



	float* Q, *K, *V, *l, *m, *Out, *OutV2;
	EcudaMalloc(&Q,   N * d * sizeof(float));
	EcudaMalloc(&K,   N * d * sizeof(float));
	EcudaMalloc(&V,   N * d * sizeof(float));
	EcudaMalloc(&l,   N *     sizeof(float));
	EcudaMalloc(&m,   N *     sizeof(float));
	EcudaMalloc(&Out, N * d * sizeof(float));

	EcudaMemcpy(Q, Qm.unsafe_data(), N * d * sizeof(float), cudaMemcpyHostToDevice);
	EcudaMemcpy(K, Km.unsafe_data(), N * d * sizeof(float), cudaMemcpyHostToDevice);
	EcudaMemcpy(V, Vm.unsafe_data(), N * d * sizeof(float), cudaMemcpyHostToDevice);
	EcudaMemcpy(l, lm.unsafe_data(), N *     sizeof(float), cudaMemcpyHostToDevice);
	EcudaMemcpy(m, mm.unsafe_data(), N *     sizeof(float), cudaMemcpyHostToDevice);
	EcudaMemset(Out, 0, N*d*sizeof(float));

	// compute kernel run time
	// warm up
	for(int i = 0; i < 10; i ++){
		kernel<<<1, Bc, sram_size>>>(Q, K, V, N, d, Tr, Tc, Br, Bc, softmax_scale, l, m, Out);
	}
	cudaEvent_t start;
	cudaEvent_t end;
	cudaEventCreate(&start);
	cudaEventCreate(&end);

	// v1
	float millisecondsum = 0;
	for(int t=0; t<10; t++){
		cudaEventRecord(start);
		kernel<<<1, Bc, sram_size>>>(Q, K, V, N, d, Tr, Tc, Br, Bc, softmax_scale, l, m, Out);
		cudaEventRecord(end);
		cudaEventSynchronize(end);
		float milliseconds = 0;
		cudaEventElapsedTime(&milliseconds, start, end);
		millisecondsum += milliseconds;
	}
	millisecondsum /= 10.f;

	EcudaMemcpy(Outm.unsafe_data(), Out, N * d * sizeof(float), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();

	// v2
	int sram_size_v2 = sram_size + (Bc * d) * sizeof(float);
	float millisecondsumV2 = 0;
	for(int t=0; t<10; t++){
		cudaEventRecord(start);
		kernelV2<<<1, Bc, sram_size_v2>>>(Q, K, V, N, d, Tr, Tc, Br, Bc, softmax_scale, Out);
		cudaEventRecord(end);
		cudaEventSynchronize(end);
		float milliseconds = 0;
		cudaEventElapsedTime(&milliseconds, start, end);
		millisecondsumV2 += milliseconds;
	}
	millisecondsumV2 /= 10.f;

	EcudaMemcpy(OutmV2.unsafe_data(), Out, N * d * sizeof(float), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();


	// compute cpu run time
	auto start_cpu = std::chrono::high_resolution_clock::now();
	for( int i = 0; i < 50; i ++){
		attention_cpu(Qm, Km, Vm, softmax_scale, Out_compare1, N, d);
	}
	auto end_cpu = std::chrono::high_resolution_clock::now();
	float attntime = std::chrono::duration_cast<std::chrono::milliseconds>(end_cpu - start_cpu).count() / 50.f;
	auto start_cpu2 = std::chrono::high_resolution_clock::now();
	for( int i = 0; i < 50; i ++){
		flashattn_cpu(Qm, Km, Vm, softmax_scale, Out_compare2, N, d);
	}
	auto end_cpu2 = std::chrono::high_resolution_clock::now();
	float flashtime = std::chrono::duration_cast<std::chrono::milliseconds>(end_cpu2 - start_cpu2).count() / 50.f;


	// error message
	float error1 = matrix<float>::diff(Outm, Out_compare1);
	float error2 = matrix<float>::diff(Outm, Out_compare2);
	float error3 = matrix<float>::diff(Out_compare1, Out_compare2);

	float error4 = matrix<float>::diff(OutmV2, Outm);

	if(error1 > 1 || error2 > 1 || error3 > 1){
		printf("flashattention forward v1\n");
		printf("  Tr: %d, Tc: %d, Br: %d, Bc: %d\n", Tr, Tc, Br, Bc);
		printf("  N: %d, d: %d, softmax_scale:%.3f\n", N, d, softmax_scale);
		int sram_size_max;
		cudaDeviceGetAttribute(&sram_size_max, cudaDevAttrMaxSharedMemoryPerBlock, 0);
		printf("  V1: sram_size_used: %d, sram_size_max: %d, max_d: %d\n", sram_size, sram_size_max, ((sram_size_max/4)-32*32)/96);
		printf("  V2: sram_size_used: %d, sram_size_max: %d, max_d: %d\n", sram_size_v2, sram_size_max, ((sram_size_max/4)-32*32)/128);
		printf("  Error: %f, %f, %f, %f\n", error1, error2, error3, error4);
	}

	return {millisecondsum, millisecondsumV2, attntime, flashtime};
}