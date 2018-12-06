#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

// Maximum value of the matrix element
#define MAX 100

#define MAX_ITER 100
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




// Write the time results into a CSV file
void write_to_file(int n, int num_blocks, int num_threads, float total_time, float exec_time) {

	FILE *f;
	char* file_name = "results.csv";

	if (access(file_name, F_OK) == -1) {
 		f = fopen(file_name, "a");
		fprintf(f, "Matrix size;Blocks;Threads per block;Total time;Operations time;\n");
	}
	else {
		f = fopen(file_name, "a");
	}

	fprintf(f, "%d;%d;%d;%f;%f;\n", n, num_blocks, num_threads, total_time, exec_time);
	fclose(f);
}




// Solver (executed by each thread)
__global__ void solver(float **mat, int n) {

	// Position this thread is going to compute
	int i = (blockDim.x * blockIdx.x) + threadIdx.x;
	i = i + n;	// VIP: The threads must avoid first row
	i = i + 1;	// VIP: The threads must avoid first column

	// In case the thread is leftover
	if (i >= ((n*n) - n - 1)) {
		return 0;
	}


	float diff = 0, temp;
	int done = 0, cnt_iter = 0;

	while (!done && (cnt_iter < MAX_ITER)) {
		const int pos_up = i - n;
		const int pos_do = i + n;
		const int pos_le = i - 1;
		const int pos_ri = i + 1;

		temp = (*mat)[i];
		(*mat)[i] = 0.2 * ((*mat)[i] + (*mat)[pos_le] + (*mat)[pos_up] + (*mat)[pos_ri] + (*mat)[pos_do]);
		diff += abs((*mat)[i] - temp);

		// TODO: Revisar porque hay que hacer uno por cada iteracion, no casilla --> llevar fuera de la gpu, tener una matriz de diffs
		if (diff/n/n < TOL) {
			done = 1;
		}
		cnt_iter ++;
	}
}




int main(int argc, char *argv[]) {

	if (argc < 3) {
		printf("Call this program with two parameters:\n");
		printf("\t matrix_size: Add 2 to a power of 2 (e.g. : 18, 1026)\n");
		printf("\t threads_per_block: Better a power of 2 (e.g. : 16, 32, 64)\n");
		exit(1);
	}

	int n = atoi(argv[1]);
	int threads_per_block = atoi(argv[2]);
	printf("Matrix size = %d\n", n);
	printf("Threads per block = %d\n", threads_per_block);


	// Start recording the time
	clock_t i_total_t = clock();


	// Calculating the memory size for allocating (bytes)
	size_t mem_size = calc_mem_size(n, n);

	float *host_mat_org, *host_mat_dest, *dev_matrix;

	// Allocating matrices space both in host and device
	allocate_host_matrix(&host_mat_org, n, n);
	allocate_host_matrix(&host_mat_dest, n, n);
	allocate_dev_matrix(&dev_mat, n, n);


	// Passing data from host to device
	cudaMemcpy(dev_mat, host_mat_org, mem_size, cudaMemcpyHostToDevice);


	// Calculate the number of threads to launch (1 per core cell)
	int core_dim = (n-2) * (n-2);

	// Given a constant number of threads per block, determine the blocks
	int num_blocks = (int) ceil(core_dim / threads_per_block);
	dim3 dimGrid(num_blocks);
	dim3 dimBlock(threads_per_block);


	// Time before the execution
	clock_t i_exec_t = clock();

	// Make all the threads synchronous
	solver<<< dimGrid, dimBlock >>>(&dev_mat, n);

	// The ThreadSynchronize would be neccesary in case Memcpy is not done
	// However, as it is called later on, the following line is commented
	// cudaThreadSynchronize();

	// Time before the execution
	clock_t f_exec_t = clock();


	// Passing data back from the device to the host
	cudaMemcpy(host_mat_dest, dev_mat, mem_size, cudaMemcpyDeviceToHost);


	// Finally, the matrices are freed
	cudaFree(dev_mat);
	free(host_mat_org);
	free(host_mat_dest);


	// Finish recording the time
	clock_t f_total_t = clock();


	float total_time = (float)(f_total_t - i_total_t) / CLOCKS_PER_SEC;
	float exec_time = (float)(f_exec_t - i_exec_t) / CLOCKS_PER_SEC;
	printf("Total time: %f\n", total_time);
	printf("Operations time: %f\n", exec_time);


	write_to_file(n, num_blocks, threads_per_block, total_time, exec_time);
	return 0;
}
