if [ ! -d build ]; then
	mkdir build
fi

cd build &&
	cmake .. && make -j8 && cd ..


# 测试指定大小 N
if [ $# -eq 1 ]; then
	./sgemv $1
	exit 0
fi

./sgemv


# for ((i=1; i<=32; i++)); do
# 	./sgemv $i
# done
