#include <iostream>
#include <fstream>
#include <vector>

__global__ void binarySearchV1(float *input, int *output, int size , float target) {
    __shared__ int found ;
    found = -1;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int step_size = ceil((float)size / blockDim.x);
    // int start = j * step_size;
    // int end = min(start + step_size, size);
    int low = j * step_size;
    int high = min(low + step_size, size) - 1;
    int mid;
    while (low <= high && found == -1) {
        mid = (low + high) / 2;
        if (input[mid] - target <= 0.0001 && input[mid] - target >= -0.0001) {
            found = mid;
        } else if (input[mid] < target) {
            low = mid + 1;
        } else {
            high = mid - 1;
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        *output = found;
    }
}
int main(int argc, char *argv[]) {
    if (argc != 3) {
        printf(" ./Q2 <input file> <target>\n");
        return 1;
    }

    std::ifstream input(argv[1]);
    if (!input) {
        printf("failed to open the file");
        return 1;
    }
    float target = atof(argv[2]);

    float x;
    std::vector<float> inputVector;
    while (input >> x) {
        inputVector.push_back(x);
    }
    int arrSize = inputVector.size();
    float *d_in; 
    int * d_out;
    int result;

    cudaMalloc(&d_in, arrSize * sizeof(float));
    cudaMalloc(&d_out, sizeof(int));
    
    cudaMemcpy(d_in, inputVector.data(), arrSize * sizeof(float), cudaMemcpyHostToDevice);
    binarySearchV1<<<1, 256>>>(d_in, d_out, arrSize, target);
    cudaMemcpy(&result, d_out, sizeof(int), cudaMemcpyDeviceToHost);
    std::cout << result ;
    
    cudaFree(d_in);
    cudaFree(d_out);
    return 0;
}
