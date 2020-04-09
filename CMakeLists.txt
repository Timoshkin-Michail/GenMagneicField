cmake_minimum_required(VERSION 3.15)
project(GenMagneticField)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_COMPILER /usr/bin/mpicxx)
set(CMAKE_C_COMPILER /usr/bin/mpic)

add_library(VectorFunction VectorFunction.cpp VectorFunction.h)

add_executable(task1 task1.cpp)
target_link_libraries(task1 libfftw3_mpi.a libfftw3.a libm.a)


add_executable(task2 task2.cpp VectorFunction)
target_link_libraries(task2 libfftw3_mpi.a libfftw3.a libm.a)
