#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <stdio.h>

// kernel3: each thread produces one output matrix column
__global__ void matrixAdd3(float* A, float* B, float* C, int rows, int cols) {
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	if (j < cols) {
		for (int i = 0; i < rows; i++) {
			C[i * cols + j] = A[i * cols + j] + B[i * cols + j];
		}
	}
}
// fix kernel 3 to work on size of 1*1000
__global__ void matrixAdd3_1_1000(float* A, float* B, float* C, int rows, int cols) {
	int j = threadIdx.y + blockIdx.y * blockDim.x;
	if (j < cols) {
		for (int i = 0; i < rows; i++) {
			C[i * cols + j] = A[i * cols + j] + B[i * cols + j];
		}
	}
}
// fix kernel 3 to work on size of 1000*1




int main(int argc, char *argv[]) {
	printf("Reading file...\n");

	const char* filename = argv[1];
  const char* outputFilename = argv[2];
	int numOfTests, rows, cols;
	
	FILE* file = fopen(filename, "r");
	FILE* outputFile = fopen(outputFilename, "w");
	
	if (file == NULL) {
		printf("Error: can't open file.\n");
		exit(1);
	}
	fscanf(file, "%d", &numOfTests);
	
	for (int i = 0; i < numOfTests; i++) {
		
		fscanf(file, "%d", &rows);
		fscanf(file, "%d", &cols);
		
		float* A = (float*)malloc(rows * cols * sizeof(float));
		float* B = (float*)malloc(rows * cols * sizeof(float));
		
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

		dim3 threadsPerBlock(32, 32);

		dim3 blocksPerGrid((int)ceil(float(rows) / threadsPerBlock.x), (int)ceil(float(cols) / threadsPerBlock.y));
		// Copy data back to host
		float* C = (float*)malloc(rows * cols * sizeof(float));
		// print matrix C with kernel 3
		matrixAdd3 << <blocksPerGrid, threadsPerBlock >> > (d_A, d_B, d_C, rows, cols);
		cudaMemcpy(C, d_C, rows * cols * sizeof(float), cudaMemcpyDeviceToHost);
		
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
		// Free memory
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
