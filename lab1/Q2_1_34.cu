
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
__global__ void matrixMul3(float* A, float* B, float* C, int rows, int cols) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	float sum = 0;
	if (i < rows) {
		for (int j = 0; j < cols; j++) {
			sum += A[i * cols + j] * B[j];
		}
		C[i] = sum;
	}
}

void REQ2(char * filename , char * outputFileName) {
      
	printf("Reading file...\n");
	//const char* filename = "test2.txt";
	int numOfTests, rows, cols;
	FILE* file = fopen(filename, "r");
	FILE* outputFile = fopen(outputFileName, "w");
	if (file == NULL) {
		printf("Error: can't open file.\n");
		exit(1);
	}
	fscanf(file, "%d", &numOfTests);
	for (int i = 0; i < numOfTests; i++) {
		fscanf(file, "%d", &rows);
		fscanf(file, "%d", &cols);
		
		float* A = (float*)malloc(rows * cols * sizeof(float));
		float* B = (float*)malloc(cols * sizeof(float));
		float* C = (float*)malloc(rows * sizeof(float));
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < cols; j++) {
				fscanf(file, "%f", &A[i * cols + j]);
			}

		}
		for (int i = 0; i < cols; i++) {
			fscanf(file, "%f", &B[i]);
		}
		// Allocate device memory
		float* d_A;
		float* d_B;
		float* d_C;
		cudaMalloc(&d_A, rows * cols * sizeof(float));
		cudaMalloc(&d_B, cols * sizeof(float));
		cudaMalloc(&d_C, rows * sizeof(float));
		// Transfer data from host to device memory
		cudaMemcpy(d_A, A, rows * cols * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(d_B, B, cols * sizeof(float), cudaMemcpyHostToDevice);
		// Kernel launch code – to be added in the next step
		// Kernel launch code
		int threadsPerBlock = 256;
		int blocksPerGrid = (rows + threadsPerBlock - 1) / threadsPerBlock;
		matrixMul3 << <blocksPerGrid, threadsPerBlock >> > (d_A, d_B, d_C, rows, cols);
		// Transfer data back to host memory
		cudaMemcpy(C, d_C, rows * sizeof(float), cudaMemcpyDeviceToHost);
		// print A
		// for (int i = 0; i < rows; i++) {
		// 	for (int j = 0; j < cols; j++) {
		// 		printf("%f ", A[i * cols + j]);
		// 	}
		// 	printf("\n");
		// }
		// // print B
		// for (int i = 0; i < cols; i++) {
		// 	printf("%f ", B[i]);
		// }
		// printf("\n");
		// print the result
		// for (int i = 0; i < rows; i++) {
		// 	printf("%f ", C[i]);
      
		// }
		if (outputFile == NULL) {
			printf("Error: can't open file.\n");
			exit(1);
		}
		for (int i = 0; i < rows; i++) {
			fprintf(outputFile, "%.1f ", C[i]);
      fprintf(outputFile, "\n");
		}
		// Free device memory
		cudaFree(d_A);
		cudaFree(d_B);
		cudaFree(d_C);
		// Free host memory
		free(A);
		free(B);
		free(C);
	}
	fclose(outputFile);

}
int main(int argc,char* argv[]) {
	char* filename = argv[1];
  char* outputFileName = argv[2];
	if (filename == NULL) {
		printf("Error: no file name provided.\n");
		exit(1);
	}
	REQ2(filename, outputFileName);
}

