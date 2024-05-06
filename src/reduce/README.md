
[ref](https://zhuanlan.zhihu.com/p/572820783)

### warp_reduce shfl_xor

![img](pic/warp_reduce.png)

具体的, warp_reduce_sum<32>为warp内同步, warp_reduce_sum<16>为前16后16分别同步.

16-31等效16, 8-15等效8, 4-7等效4, 2-3等效2

### block_reduce

![img](pic/block_reduce.png)