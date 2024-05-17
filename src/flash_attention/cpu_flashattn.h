#pragma once
#include "matrix.h"
#include <numeric>

template < class T >
void flashattn_cpu(matrix<T> &Q, matrix<T> &K, matrix<T> &V, float softmax_scale, matrix<T> &out, int N, int d){
	// traverse Q
	for ( int l = 0; l < N; l ++ ){
		
		T maxn = std::numeric_limits<T>::min();
		T sum  = 0;
		T oi[d];
		memset(oi, 0, d * sizeof(T));
		//traverse KV
		for ( int i = 0; i < N; i ++ ){
			T tmp = 0;
			// QK^T
			for ( int x = 0; x < d; x ++ ){
				tmp += Q[l*d+x] * K[i*d+x];
			}
			tmp *= softmax_scale;

			T max_prev = maxn;
			maxn = max(maxn, tmp);
			T sum_prev = sum;
			sum  = sum * exp(max_prev - maxn) + exp(tmp - maxn);

			T t1 = (sum_prev / sum) * exp(max_prev - maxn);
			T t2 = (exp(tmp - maxn) / sum);
			for ( int x = 0; x < d; x ++ ){
				oi[x] = oi[x] * t1 + t2 * V[i*d+x];
			}
		}
		for ( int x = 0; x < d; x ++ ){
			out[l*d+x] = oi[x];
		}
	}
}

template < class T >
void attention_cpu(matrix<T> &Q, matrix<T> &K, matrix<T> &V, float softmax_scale, matrix<T> &out, int N, int d){
	matrix<T> S(N, N, true, false);

	T maxn[N];
	memset(maxn, 0, sizeof(T) * N);

	for ( int i = 0; i < N; i ++ ){
		for ( int j = 0; j < N; j ++ ){
			T tmp = 0;
			for ( int x = 0; x < d; x ++ ){
				tmp += Q[i*d+x] * K[j*d+x];
			}
			tmp *= softmax_scale;

			maxn[i] = max(maxn[i], tmp);
			S[i*N+j] = tmp;
		}
	}
	// S.show("QK^T");
	// for(int i=0;i<N;i++){
	// 	printf("%f-", maxn[i]);
	// }
	// printf("\n");
	T sum[N];
	memset(sum, 0, sizeof(T) * N);
	for ( int i = 0; i < N; i ++ ){
		for ( int j = 0; j < N; j ++ ){
			T tmp = exp(S[i*N+j] - maxn[i]);
			sum[i] += tmp;
			S[i*N+j] = tmp;
		}
	}
	// S.show("e^{xi-maxi}");

	for ( int i = 0; i < N; i ++ ){
		for ( int j = 0; j < N; j ++ ){
			S[i*N+j] /= sum[i];
		}
	}
	// S.show("s");

	out = S * V;
}