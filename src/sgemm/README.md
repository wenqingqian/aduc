# aduc
.........

## TODO

in v6_hidegemmlatency.cu in V100
```cpp
if (mValid && nValid) {
	aduc::float4 result{c[a + i * layoutThread::M][b]};
	if (beta != 0) {
		result = result + pC(a, b * tileSharedIntervalBT) * beta;
	}
	pC(a, b * tileSharedIntervalBT) = result;
}
```
will have 10% improvement than below... (because of occupancy is dropped from 25% to 12.5% and reg from 128 to 132)
```cpp
if (mValid && nValid) {
	aduc::float4 result{c[a + i * layoutThread::M][b]};
	if (beta != 0) {
		result = result + pC(a, b * tileSharedIntervalBT) * beta;
		pC(a, b * tileSharedIntervalBT) = result;
	}
}
// or
if (mValid && nValid) {
	aduc::float4 result{c[a + i * layoutThread::M][b]};
	result = result + pC(a, b * tileSharedIntervalBT) * beta;
	pC(a, b * tileSharedIntervalBT) = result;
}
```

## other (maybe useless)

unused link
[Report PDF](https://wenqingqian.github.io/assets/pdf/gemm.pdf)

![img](pic/final.png)

**Tesla P4**
![img](pic/echarts.png)

