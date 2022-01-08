#############################################################
# Makefile for project, Accelerated programming. Niklas Wik #
# g++ (GCC) 8.3.0        ///   nvcc V10.1.243               #
# Intel Xeon E5-2660    ///    Nvidia Tesla T4              #
# Run with eg:                                              #
# ./app_host -N 32 -matrix scs -number double -repeat 200   #
#############################################################

CC = g++
CFLASG = -O3 -march=sandybridge -fopenmp
LIBS = -lm

NV = nvcc
NVFLAGS = -arch=sm_75

BIN = app_host app_device

all: $(BIN)

app_host: fem_application.cpp
	$(CC) $(CFLAGS) -o $@ $< $(LIBS)

app_device: fem_application.cu
	$(NV) $(NVFLAGS) -o $@ $< $(LIBS)