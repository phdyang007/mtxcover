test.bin: MatrixCover.o TestMC.o
	g++ MatrixCover.o TestMC.o -o $@

MatrixCover.o: MatrixCover.cpp
	g++ -c -o $@ $<

TestMC.o: TestMC.cpp
	g++ -c -o $@ $<