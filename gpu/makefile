
test.bin: MatrixCoverGPU.o TestMC.o
	nvcc MatrixCoverGPU.o TestMC.o -o $@

MatrixCoverGPU.o: MatrixCoverGPU.cu 
	nvcc -c MatrixCoverGPU.cu

TestMC.o: TestMC.cu 
	nvcc -c TestMC.cu