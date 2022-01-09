###############################################################################
# Makefile for project, Parallel and Distributed Computing 2021. Niklas Wik
###############################################################################

CC = mpicc
CFLAGS = -std=c99 -g -Wall -O3
LIBS = -lm

BIN = shear_sort shear_sort_nm


all: $(BIN)

shear_sort: shear_sort.c
	$(CC) $(CFLAGS) -o $@ $< $(LIBS)

shear_sort_nm: shear_sort_nm.c
	$(CC) $(CFLAGS) -o $@ $< $(LIBS)

clean:
	$(RM) $(BIN)
