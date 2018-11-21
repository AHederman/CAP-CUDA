CC      = nvcc
LD      = $(CC)

CFLAGS  = -g -Wall
CFLAGS  += -I.
LDFLAGS =

OBJS    =
LIBS    = -lm

gs_cuda: gs_cuda.o
        g++ -o gs_cuda gs_cuda.o -L/home/software/nvidia/cuda-8.0/lib64/  -lcudart -lcublas -fopenmp -O3 -Wextra -std=c++11

gs_cuda.o: gs_cuda.cu
        nvcc -std=c++11 -c -arch=sm_35 gs_cuda.cu

clean:
        -rm -f *.o  *~ $(PRGS)
