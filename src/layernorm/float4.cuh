#pragma once

namespace aduc{

struct __device_builtin__ __builtin_align__(16) float4{
	float data_[4];

	__host__ __device__ float &operator[](unsigned idx){
		return data_[idx]; 
	}

	__host__ __device__ float operator[](unsigned idx) const {
		return data_[idx];
	}

	__host__ __device__ float4 operator+(const float4 &a){
		return float4{data_[0]+a.data_[0], data_[1]+a.data_[1],
						data_[2]+a.data_[2], data_[3]+a.data_[3]};
	}

	__host__ __device__ float4 operator*(float f) const{
		return float4{data_[0]*f, data_[1]*f, data_[2]*f, data_[3]*f};
	}

	__host__ __device__ float4 operator/(float f) const{
		return float4{data_[0]/f, data_[1]/f, data_[2]/f, data_[3]/f};
	}

	__host__ __device__ float operator*(const float4 &f) const{
		return data_[0]*f[0] + data_[1]*f[1] + data_[2]*f[2] + data_[3]*f[3];
	}
};

}

#define f4 aduc::float4