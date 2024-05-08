#pragma once
#include <cuda_runtime.h>

namespace aduc{
	
__device__ __inline__ float atomicMaxFloat(float* address, float val) {
	int* address_as_int = reinterpret_cast<int*>(address);
	int old = *address_as_int, assumed;
	do {
		assumed = old;
		old = atomicCAS(address_as_int, assumed, __float_as_int(fmaxf(val, __int_as_float(assumed))));
	} while (assumed != old);
	return __int_as_float(old);
}

}