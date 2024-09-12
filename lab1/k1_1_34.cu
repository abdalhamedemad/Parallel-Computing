#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <stdio.h>

// Kernel 1: Each thread produces one output matrix element
__global__ void matrixAdd(float* A, float* B, float* C, int rows, int cols) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	if (i < rows && j < cols) {
		C[i * cols + j] = A[i * cols + j] + B[i * cols + j];
	}
}

int main(int argc, char *argv[]) {
	printf("Reading file...\n");
	// Read input file
	const char* filename = argv[1];
	// Read output file
  const char* outputFilename = argv[2];
	int numOfTests, rows, cols;
	// open file for writing
	FILE* outputFile = fopen(outputFilename, "w");
	// open file for reading
	FILE* file = fopen(filename, "r");
	if (file == NULL) {
		printf("Error: can't open file.\n");
		exit(1);
	}
	// read number of tests
	fscanf(file, "%d", &numOfTests);

	for (int i = 0; i < numOfTests; i++) {
		// read rows and cols
		fscanf(file, "%d", &rows);
		fscanf(file, "%d", &cols);
		// read matrix A and B
		float* A = (float*)malloc(rows * cols * sizeof(float));
		float* B = (float*)malloc(rows * cols * sizeof(float));
		// read matrix A and B
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < cols; j++) {
				fscanf(file, "%f", &A[i * cols + j]);
			}
		}
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < cols; j++) {
				fscanf(file, "%f", &B[i * cols + j]);
			}
		}

		// Allocate memory on the device
		float* d_A, * d_B, * d_C;
		cudaMalloc(&d_A, rows * cols * sizeof(float));
		cudaMalloc(&d_B, rows * cols * sizeof(float));
		cudaMalloc(&d_C, rows * cols * sizeof(float));

		// Copy data to the device
		cudaMemcpy(d_A, A, rows * cols * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(d_B, B, rows * cols * sizeof(float), cudaMemcpyHostToDevice);

		// threadsPerBlock and blocksPerGrid
		dim3 threadsPerBlock(32, 32);
		dim3 blocksPerGrid((int)ceil(float(rows) / threadsPerBlock.x), (int)ceil(float(cols) / threadsPerBlock.y));

		// Launch addKernel() on GPU
		matrixAdd << <blocksPerGrid, threadsPerBlock >> > (d_A, d_B, d_C, rows, cols);

		// Copy data back to host
		float* C = (float*)malloc(rows * cols * sizeof(float));
		cudaMemcpy(C, d_C, rows * cols * sizeof(float), cudaMemcpyDeviceToHost);

		// print output to file
		printf("Writing to file...\n");
		if (outputFile == NULL) {
			printf("Error: can't open file.\n");
			exit(1);
		}
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < cols; j++) {
				fprintf(outputFile, "%.1f ", C[i * cols + j]);
				//fprintf(outputFile, "%f ", C[i * cols + j]);
			}
			fprintf(outputFile, "\n");
		}
		printf("Done writing to file.\n");
		free(A);
		free(B);
		free(C);
		cudaFree(d_A);
		cudaFree(d_B);
		cudaFree(d_C);
	}
	fclose(file);
	fclose(outputFile);
	return 0;
}

