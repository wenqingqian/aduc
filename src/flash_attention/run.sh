if [ ! -d build ]; then
	mkdir build
fi

cd build &&
	cmake .. && make -j8 && cd ..

./flash_attn