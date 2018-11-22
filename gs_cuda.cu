#include <stdio.h>
#include <stdlib.h>
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




// Solver (TODO)
__global__ void solver(float **mat, int n, int m, int num_elems) {

	float diff = 0, temp;
	int done = 0, cnt_iter = 0, i, j;

	while (!threadIdx.done && (threadIdx.cnt_iter < MAX_ITER)) {
		diff = 0;
		
		// No termino de tener muy claro que esto sea asi, pero :D
		for (i = 1; i < n - 1; i++) {
  			for (j = 1; j < m - 1; j++) {
				(*mat)[threadIdx.i*m + threadIdx.j] = 0.2 * ((*mat)[threadIdx.i*m + threadIdx.j] + 
									     (*mat)[threadIdx.i*m + (threadIdx.j - 1)] + 
									     (*mat)[(threadIdx.i - 1)*m + threadIdx.j] + 
									     (*mat)[threadIdx.i*m + (threadIdx.j + 1)] + 
									     (*mat)[(threadIdx.i + 1)*m + threadIdx.j]);
				threadIdx.diff += abs((*mat)[threadIdx.i*m + threadIdx.j] - threadIdx.temp);
		
			}
		}
		/*for (i = 1; i < n - 1; i++) {
			for (j = 1; j < m - 1; j++) {
				temp = (*mat)[i][j];
				(*mat)[i][j] = 0.2 * ((*mat)[i][j] + (*mat)[i][j - 1] + (*mat)[i - 1][j] + (*mat)[i][j + 1] + (*mat)[i + 1][j]);
				diff += abs((*mat)[i][j] - temp);
			}
		}*/

		if (threadIdx.diff/n/n < TOL) {
			threadIdx.done = 1; 
		}
		threadIdx.cnt_iter ++;
	}

	if (threadIdx.done) {
		printf("Solver converged after %d iterations\n", threadIdx.cnt_iter);
	}
	else {
		printf("Solver not converged after %d iterations\n", threadIdx.cnt_iter);
	}
}




int main(int argc, char *argv[]) {
	// Primera medida sobre el tiempo total
	clock_t ttime, extime;
	ttime = clock();
	
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
	
	// define grid and block size
	int numBlocks = 8;
	int numThreadsPerBlock = 8;

	// Configure and launch kernel
	dim3 dimGrid();
	dim3 dimBlock();
	
	// Medida sobre el tiempo de ejecucion
	extime = clock();
	
	solver<<<dimGrid,dimBlock>>>(&dev_mat, n, n, num_elems);
	
	// Fin del tiempo de ejecucion
	extime = clock() - extime;
	
	// block until the device has completed
	cudaThreadSynchronize();
	*/


	// Passing data back from the device to the host
	cudaMemcpy(host_mat_dest, dev_mat, mem_size, cudaMemcpyDeviceToHost);

	// Finally, the matrices are freed
	cudaFree(dev_mat);
	free(host_mat_org);
	free(host_mat_dest);
	
	// Fin del tiempo total
	ttime = clock() - ttime;
	
	//Mensaje final e imprimir en fichero
	printf("Total time: %f\n", ttime);
	printf("Operations time: %f\n", extime);

	FILE *f;
	if (access("results.csv", F_OK) == -1) {
 		f = fopen("results.csv", "a");
		fprintf(f, "Matrix size;Block size;Total time;Size;Operations time;\n");
	}
	else {
		f = fopen("results.csv", "a");
	}

	fprintf(f, "%d;%d;%f;%f;\n", n, dimBlock(), ttime, extime);
	fclose(f);

	
	return 0;
}
