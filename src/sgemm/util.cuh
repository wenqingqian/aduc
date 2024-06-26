#pragma once

#define def_gemm(name) \
	void name(const float *A, const float *B, float *C, \
		float alpha, float beta, unsigned M, unsigned N, unsigned K) 

#define extern_gemm(name)\
	extern def_gemm(name);

namespace aduc{
#define CUDAF2 __host__ __device__

template < class T >
struct __device_builtin__ tensor2d {
	T * ptr_;
	const unsigned row_size_, col_size_;
	int row_offset_{0};
	int col_offset_{0};

	template<class T2>
	CUDAF2 tensor2d(T2* p, unsigned row, unsigned col):
		ptr_(reinterpret_cast<T*>(p)), row_size_(row), col_size_(col)
		{}
	
	CUDAF2 void addOffset(int row, int col){
		row_offset_ += row;
		col_offset_ += col;
	}

	CUDAF2 bool isRowValid(int offset = 0) const {
		return (row_offset_ + offset) < row_size_;
	}

	CUDAF2 bool isColValid(int offset = 0) const {
		return (col_offset_ + offset) < col_size_;
	}

	CUDAF2 bool isValid(){
		return row_offset_ < row_size_ && col_offset_ < col_size_;
	}

	CUDAF2 T& operator()(int row, int col){
		return ptr_[(row + row_offset_) * col_size_ + col + col_offset_];
	}

};

struct __device_builtin__ __builtin_align__(16) float4{
	float data_[4];

	CUDAF2 float &operator[](unsigned idx){
		return data_[idx]; 
	}

	CUDAF2 float operator[](unsigned idx) const {
		return data_[idx];
	}

	CUDAF2 float4 operator+(const float4 &a){
		return float4{data_[0]+a.data_[0], data_[1]+a.data_[1],
					  data_[2]+a.data_[2], data_[3]+a.data_[3]};
	}

	CUDAF2 float4 operator*(float f) const{
		return float4{data_[0]*f, data_[1]*f, data_[2]*f, data_[3]*f};
	}
};

template < unsigned M_ = 1, unsigned N_ = 1, unsigned K_ = 1>
struct layout {
	static constexpr unsigned M = M_;
	static constexpr unsigned N = N_;
	static constexpr unsigned K = K_;
};


}