all: cpu.bin gpumg.bin

#target_fn.a: common.h target_fn.cu target_fn.h ../cpu/MatrixCover.h ../gpu/MatrixCoverGPU.cuh ../gpu-mg/MatrixCoverGPU.cuh
#	mkdir -p build
#	nvcc -c -O2 -lineinfo -arch=sm_35 -std=c++11 -rdc=false -Xcompiler -fopenmp \
#		target_fn.cu -o build/target_fn.o -DBENCHMARK 
#	ar cr build/target_fn.a \
#		build/target_fn.o 

cpu.bin: benchmark.cu
	mkdir -p build
	nvcc -O2 -lineinfo -arch=sm_35 -std=c++11 -rdc=false -I. \
		  -DBENCHMARK \
		 -lcudadevrt -Xcompiler -fopenmp benchmark.cu -DCPU -o  build/$@


gpumg.bin: benchmark.cu
	mkdir -p build
	nvcc -O2 -std=c++11 \
		  -DBENCHMARK \
		  benchmark.cu \
		  -lcudart -DGPUMG -o build/$@


clean:
	rm -rf build
