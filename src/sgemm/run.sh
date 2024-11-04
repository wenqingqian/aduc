if [ ! -d build ]; then
	mkdir build
fi

filename="output.txt"

if [ "$1" == "clean" ]; then
	# 检查文件是否存在并且不为空
	if [ -s "$filename" ]; then
		# 清空文件内容
		> "$filename"
	fi

	# 在文件开头添加 'aaa' 并将光标移到下一行
	echo -e "name,M,N,K,alpha,beta,error,time,gflops,compare_to_cublas,compare_to_peak\n" > "$filename"
fi

# 初始值
start=256
# 步长
step=256
# 结束值
end=7680

# 进入build目录
cd build && \
	cmake -DSTORE_RESULT=on \
		  -DDISABLE_CPU=on \
		  .. \
	&& make -j8 && cd ..

# 循环执行gemm命令，从start到end，步长为step
for i in $(seq $start $step $end); do
	echo "begin $i"
	./gemm $i $i $i
done