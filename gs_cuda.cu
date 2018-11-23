#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>


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




// Solver (executed by each thread)
__global__ void solver(float **mat, int n) {

	// Position this thread is going to compute
	int i = (blockDim.x * blockIdx.x) + threadIdx.x;
	i = i + n;	// VIP: The threads must avoid first row
	i = i + 1;	// VIP: The threads must avoid first column

	// In case the thread is leftover
	if (i > ((n*n) - n - 2)) {
		return 0;
	}


	float diff = 0, temp;
	int done = 0, cnt_iter = 0;

	while (!done && (cnt_iter < MAX_ITER)) {
		diff = 0;

		int pos_up = i - n;
		int pos_do = i + n;
		int pos_le = i - 1;
		int pos_ri = i + 1;

		temp = (*mat)[i];
		(*mat)[i] = 0.2 * ((*mat)[i] + (*mat)[pos_le] + (*mat)[pos_up] + (*mat)[pos_ri] + (*mat)[pos_do]);
		diff += abs((*mat)[i] - temp);

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

	// Start recording the time
	clock_t total_time = clock();

	float *host_mat_org, *host_mat_dest, *dev_matrix;

	if (argc < 2) {
		printf("Call this program with two parameters: matrix_size communication\n");
		printf("\t matrix_size: Add 2 to a power of 2 (e.g. : 18, 1026)\n");
		exit(1);
	}

	int n = atoi(argv[1]);
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
	//////////////////////// COSAS QUE HA DICHO EL PROFE ////////////////////////
	// Dejar el tamaño del bloque fijo.
		No se pueden crear bloques mayores a 1024x1024
		Para hacer pruebas, cambiar el numero de bloques para ver como varian los resultados
		Para hacer esto esta bien parametrizarlo para recibirlo como input a la hora de hacer las pruebas
		Para evitar tener que estar cambiando el fichero en cada ejecucion y eso.
		Como nos dijo el otro dia, intentar siempre que sea divisible entre 32
		Al depurar el kernel no se pueden poner prints para depurar, lo quitaron a partir de la version 8. 
		Consejo: tener un vector de control que se pueda copiar al host para poder trazar errores y esas cosas.
		Si no converge, se puede cambiar el factor TOL para que converja, lo interesante es ver que hace CUDA no que 
		fufe el algoritmo perfectamente con una convergencia que cuadre.
	//////////////////////////////////////////////////////////////////////////////
	*/


	// TODO: Adjust
	int core_dim = (n-2) * (n-2);

	// Given a constant number of threads per block, determine the blocks
	int numThreadsPerBlock = 32;
	int numBlocks = (int) ceil(core_dim / numThreadsPerBlock);

	dim3 dimGrid(numBlocks);
	dim3 dimBlock(numThreadsPerBlock);


	// Time before the execution
	clock_t exec_time = clock();

	// Make all the threads synchronous
	solver<<< dimGrid, dimBlock >>>(&dev_mat, n);
	cudaThreadSynchronize();

	// Time after the execution
	exec_time = clock() - exec_time;


	// Passing data back from the device to the host
	cudaMemcpy(host_mat_dest, dev_mat, mem_size, cudaMemcpyDeviceToHost);

	// Finally, the matrices are freed
	cudaFree(dev_mat);
	free(host_mat_org);
	free(host_mat_dest);


	// Finish recording the time
	total_time = clock() - total_time;

	// Data to a file
	printf("Total time: %f\n", total_time);
	printf("Operations time: %f\n", exec_time);

	FILE *f;
	if (access("results.csv", F_OK) == -1) {
 		f = fopen("results.csv", "a");
		fprintf(f, "Matrix size;Block size;Threads Number;Total time;Operations time;\n");
	}
	else {
		f = fopen("results.csv", "a");
	}

	fprintf(f, "%d;%d;%d;%f;%f;\n", n, numBlocks, numThreadsPerBlock, total_time, exec_time);
	fclose(f);
	return 0;
}
