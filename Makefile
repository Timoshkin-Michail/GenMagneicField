FLAGS		=	-O2 -std=c++11 -Wall
FFTW_FLAGS 	=	-lfftw3_mpi -lfftw3 -lm
COMPILER = mpic++

all: task2 clean

clean:
	rm -f task2.o VectorFunction.o

clear: clean
	rm -f task2

task2: task2.o VectorFunction.o
	$(COMPILER) -o task2 task2.o VectorFunction.o $(FLAGS) $(FFTW_FLAGS)
task2.o: task2.cpp
	$(COMPILER) -o task2.o -c task2.cpp
VectorFunction.o: VectorFunction.cpp
	$(COMPILER) -o VectorFunction.o -c VectorFunction.cpp