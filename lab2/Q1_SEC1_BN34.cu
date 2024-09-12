// %%writefile Q1.cu

#include <iostream>
#include <fstream>
#include <vector>

#define BLOCK_SIZE 256

__global__ void sumArrayV1(float *input, float *output, int size) {
    __shared__ float sum_vector[BLOCK_SIZE];
    
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int step_size = ceil((float)size / blockDim.x);
    int start = j * step_size;
    int end = min(start + step_size, size);
    for (int k = start; k < end; k++) {
        sum_vector[threadIdx.x] += input[k];
    }
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sum_vector[threadIdx.x] += sum_vector[threadIdx.x + s];
        }
        __syncthreads();
    }
    
    if (threadIdx.x == 0) {
        output[blockIdx.x] = sum_vector[0];
    }

}


int main(int argc, char *argv[]) {
    if (argc != 2) {
        printf(" ./Q1 <input file> <target>\n");
        return 1;
    }

    std::ifstream input(argv[1]);
    if (!input) {
        printf("failed to open the file");
        return 1;
    }

    float temp;
    std::vector<float> inputVector;
    while (input >> temp) {
        inputVector.push_back(temp);
    }
    int size = inputVector.size();

    float *d_in, *d_out;
    cudaMalloc((void **)&d_in, size * sizeof(float));
    cudaMalloc((void **)&d_out, sizeof(float));

    cudaMemcpy(d_in, inputVector.data(), size * sizeof(float), cudaMemcpyHostToDevice);

    sumArrayV1<<<1, BLOCK_SIZE>>>(d_in, d_out, size);

    float output;
    cudaMemcpy(&output, d_out, sizeof(float), cudaMemcpyDeviceToHost);
    printf("%f", output);

    cudaFree(d_in);
    cudaFree(d_out);

    return 0;
}
