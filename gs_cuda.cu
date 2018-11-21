#include <stdio.h>
#include <stdlib.h>


#define MAX_ITER 100

// Maximum value of the matrix element
#define MAX 100
#define TOL 0.000001




// Generate a random float number with the maximum value of max
float rand_float(int max) {
	return ((float)rand() / (float)(RAND_MAX)) * max;
}




// Calulates the space for the matrices (bytes)
int calc_mem_size(int n, int m) {
	return (n * m * sizeof(float));
}




// Allocate 2D matrix in the host
void allocate_host_matrix(float **mat, int n, int m) {

	*mat = (float *) malloc(n * m * sizeof(float));
	for (int i = 0; i < (n*m); i++) {
		(*mat)[i] = rand_float(MAX);
	}
}




// Allocate 2D matrix in the device
void allocate_dev_matrix(float **mat, int n, int m) {
	size_t memSize = (n * m * sizeof(float));
	cudaMalloc(&mat, memSize);
}




// Solver (TODO)
__global__ void solver(float **mat, int n, int m) {

	float diff = 0, temp;
	int done = 0, cnt_iter = 0, i, j;

	while (!done && (cnt_iter < MAX_ITER)) {
		diff = 0;

		for (i = 1; i < n - 1; i++) {
			for (j = 1; j < m - 1; j++) {
				temp = (*mat)[i][j];
				(*mat)[i][j] = 0.2 * ((*mat)[i][j] + (*mat)[i][j - 1] + (*mat)[i - 1][j] + (*mat)[i][j + 1] + (*mat)[i + 1][j]);
				diff += abs((*mat)[i][j] - temp);
			}
		}

		if (diff/n/n < TOL) {
			done = 1; 
		}
		cnt_iter ++;
	}

	if (done) {
		printf("Solver converged after %d iterations\n", cnt_iter);
	}
	else {
		printf("Solver not converged after %d iterations\n", cnt_iter);
	}
}




int main(int argc, char *argv[]) {

	int n, communication;
	float *host_mat_org, *host_mat_dest, *dev_matrix;

	if (argc < 2) {
		printf("Call this program with two parameters: matrix_size communication\n");
		printf("\t matrix_size: Add 2 to a power of 2 (e.g. : 18, 1026)\n");
		exit(1);
	}

	n = atoi(argv[1]);
	printf("Matrix size = %d\n", n);


	// Calculating the memory size for allocating (bytes)
	size_t mem_size = calc_mem_size(n, n);

	// Allocating matrices space both in host and device
	allocate_host_matrix(&host_mat_org, n, n);
	allocate_host_matrix(&host_mat_dest, n, n);
	allocate_dev_matrix(&dev_mat, n, n);


	// Passing data from host to device
	cudaMemcpy(dev_mat, host_mat_org, mem_size, cudaMemcpyHostToDevice);


	/*
	// define grid and block size
	int numBlocks = 8;
	int numThreadsPerBlock = 8;

	// Configure and launch kernel
	dim3 dimGrid();
	dim3 dimBlock();
	solver<<<dimGrid,dimBlock>>>(&dev_mat, n, n);

	// block until the device has completed
	cudaThreadSynchronize();
	*/


	// Passing data back from the device to the host
	cudaMemcpy(host_mat_dest, dev_mat, mem_size, cudaMemcpyDeviceToHost);

	// Finally, the matrices are freed
	cudaFree(dev_mat);
	free(host_mat_org);
	free(host_mat_dest);

	return 0;
}
