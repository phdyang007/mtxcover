
test.bin: MatrixCoverGPU.o TestMC.o
	nvcc -arch=sm_35 -rdc=true MatrixCoverGPU.o TestMC.o -lcudadevrt -o $@

MatrixCoverGPU.o: MatrixCoverGPU.cu 
	nvcc -c -arch=sm_35 -rdc=true MatrixCoverGPU.cu

TestMC.o: TestMC.cu 
	nvcc -c -arch=sm_35 -rdc=true TestMC.cu

	