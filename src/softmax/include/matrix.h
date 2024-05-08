#pragma once
#include <random>
#include <stdio.h>
#include <type_traits>
#include <assert.h>

template < class T >
class matrix{
public:
	matrix(int m, int n, bool if_alloc = true, bool if_initial = true): M(m), N(n) {
		if (if_alloc == false && if_initial == true){
			printf("warning: wrong construct while if_alloc=false & if_initial=true\n");
		}
		if(if_alloc){
			assert( data == nullptr );
			data = new T[m*n];
			if(if_initial)
				setRandom<T>();
		}
	}
	~matrix(){
		if(data != nullptr)
			free(data);
	}
	matrix(const matrix& ma): M(ma.M), N(ma.N){
		// 只能用于初始化时拷贝构造
		assert(data == nullptr);

		data = new T[M * N];
		for( int i = 0; i < M; i ++ ){
			for( int j = 0; j < N; j ++ ){
				data[i*N+j] = ma[i*N+j];
			}
		}
	}
	matrix& operator = (const matrix& ma){
		assert(data == nullptr);
		assert(M == ma.M && N == ma.N);

		data = new T[M * N];
		for( int i = 0; i < M; i ++ ){
			for( int j = 0; j < N; j ++ ){
				data[i*N+j] = ma[i*N+j];
			}
		}
		return *this;
	}

	matrix(matrix&& ma): M(ma.M), N(ma.N){
		if(data) free(data);

		data = ma.data;
		ma.data = nullptr;
	}
	
	matrix& operator = (matrix && ma){
		if(data) free(data);

		data = ma.data;
		ma.data = nullptr;
		return *this;
	}

	template< typename Ts >
	std::enable_if_t<std::is_same<Ts,int32_t>::value || std::is_same<Ts,int64_t>::value> 
	setRandom(){
		std::random_device rd;
    	std::mt19937 gen(rd());
		std::uniform_int_distribution<T> dist(1, 10);

		for(int i=0;i<M;i++){
			for(int j=0;j<N;j++){
				data[i*N+j] = dist(gen);
			}
		}
	}
	template< typename Ts >
	std::enable_if_t<std::is_same<Ts,float>::value || std::is_same<Ts,double>::value> 
	setRandom(){
		std::random_device rd;
    	std::mt19937 gen(rd());
		std::uniform_real_distribution<T> dist(1, 10);

		for(int i=0;i<M;i++){
			for(int j=0;j<N;j++){
				data[i*N+j] = dist(gen);
			}
		}
	}

	T& operator [](int idx){
		return data[idx];
	}

	T operator [](int idx) const{
		return data[idx];
	}

	matrix operator * ( matrix& ma) {
		// printf("%d-%d-%d-%d\n",M,N,ma.M,ma.N);
		assert( N == ma.M );
		matrix<T> nma(M, ma.N, true, false);
		for(int i = 0; i < M; i ++){
			for(int j = 0; j < ma.N; j ++){
				T tmp = 0;
				for(int a = 0; a<N; a++){
					tmp += data[i*N+a] * ma[a*ma.N+j];
				}
				nma[i*ma.N+j] = tmp;
			}
		}
		return nma;
	}

	static T diff(matrix& ma1, matrix& ma2){
		assert(ma1.M == ma2.M);
		assert(ma1.N == ma2.N);
		T ans = 0;
		for(int i = 0; i < ma1.M; i++){
			for(int j = 0; j < ma1.N; j++){
				ans += abs(ma1[i*ma1.N+j]-ma2[i*ma1.N+j]);
			}
		}
		return ans;
	}

	void show(){
		if (std::is_same<T, int>::value || std::is_same<T, int64_t>::value){
			printf("--------\n");
			for(int i=0;i<M;i++){
				printf("[");
				for(int j=0;j<N;j++){
					printf("%d,",data[i*N+j]);
				}
				printf("\b]\n");
			}
			printf("--------\n");
		}else{
			printf("--------\n");
			for(int i=0;i<M;i++){
				printf("[");
				for(int j=0;j<N;j++){
					printf("%f,",data[i*N+j]);
				}
				printf("\b]\n");
			}
			printf("--------\n");
		}
	}

	T* unsafe_data(){
		return data;
	}
public:
	const int M, N;
private:
	T* data = nullptr;
};