all: cpu.bin gpu.bin gpumg.bin

basic.o: common.h data_reader.h data_reader.cc
	mkdir -p build
	gcc -c -O2 -std=c++11 data_reader.cc -o build/basic.o

target_fn.a: common.h target_fn.cu target_fn.h ../gpu/MatrixCoverGPU.cu ../gpu-mg/MatrixCoverGPU.cu
	mkdir -p build
	nvcc -c -O2 -lineinfo -arch=sm_35 -std=c++11 -rdc=false -I../gpu \
		../gpu/MatrixCoverGPU.cu -DBENCHMARK -o build/gpu_cover.o
	nvcc -c -O2 -lineinfo -arch=sm_35 -std=c++11 -rdc=false -I../gpu-mg \
		../gpu-mg/MatrixCoverGPU.cu -DBENCHMARK -o build/gpu_mg_cover.o
	nvcc -c -O2 -lineinfo -arch=sm_35 -std=c++11 -rdc=false \
		target_fn.cu -o build/target_fn.o
	gcc -c -O2 -std=c++11 -I../cpu ../cpu/MatrixCover.cpp \
		-DBENCHMARK -o build/cpu_matrix_cover.o
	ar cr build/target_fn.a build/gpu_mg_cover.o \
		build/gpu_cover.o build/target_fn.o \
		build/cpu_matrix_cover.o

cpu.bin: basic.o target_fn.a
	mkdir -p build
	g++ -O2 -std=c++11 \
		 benchmark.cc build/target_fn.a build/basic.o	\
		 -lcudart -fopenmp -DCPU -o  build/$@

gpu.bin: basic.o target_fn.a
	mkdir -p build
	g++ -O2 -std=c++11 \
		  benchmark.cc build/target_fn.a build/basic.o	\
		  -lcudart -DGPU -o build/$@

gpumg.bin: basic.o target_fn.a
	mkdir -p build
	g++ -O2 -std=c++11 \
		  benchmark.cc build/target_fn.a build/basic.o	\
		  -lcudart -DGPUMG -o build/$@


clean:
	rm -rf build
