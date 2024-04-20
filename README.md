# aduc
.........

[profile](https://wenqingqian.github.io/assets/html/resource/assets/aduc.html)

![img](pic/echarts.png)

1. (v2)gemmTile

![img](pic/v2.png)

一个线程负责计算红色区域的数据

2. (v3)gemmShareMem

读取数据到共享内存时:(一个线程跨步同时处理4个float4, AB各两个)

A
![img](pic/v3.png)

B
![img](pic/v3_r.png)

计算时:

![img](pic/v3_cal_2.png)

一个线程计算 8x8 的数据, 这些数据是分散的需要映射到矩阵C上

计算时和读取内存到共享内存时线程排布不一致, 计算时类似v2, 增加了计算强度

3. (v3)gemmShareMemECG

[blog](https://zhuanlan.zhihu.com/p/531498210)中对于读取B到共享内存是把B沿行方向剪开

![img](pic/v3_cutB.png)

实测速度会慢13%PF, 在nsys report中看到寄存器使用量从128增加到了143


reference:

1. OpenMLSys
	- [blog](https://zhuanlan.zhihu.com/p/531498210)
	- [github](https://github.com/openmlsys/openmlsys-cuda/tree/main)