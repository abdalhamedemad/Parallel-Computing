#include "cuda_runtime.h"
#define MAX_NODES 1000
#include <iostream>
#include <fstream>
#include <vector>
#include "utils1.h"

using namespace std;

__global__
void bfs_kernel_top_down(unsigned int *d_row, unsigned int *d_col, unsigned int *d_row_size, unsigned int *d_col_size, unsigned int *d_level, unsigned int *d_new_vertex_visited, unsigned int *d_current_level, unsigned int *num_of_nodes) {
    unsigned int vertex = blockIdx.x * blockDim.x + threadIdx.x;
    if (vertex < *num_of_nodes) {
        if (d_level[vertex] == *d_current_level - 1) {
            for (unsigned int i = d_row[vertex]; i < d_row[vertex + 1]; ++i) {
                if (d_level[d_col[i]] == UINT_MAX) {
                    d_level[d_col[i]] = *d_current_level;
                    *d_new_vertex_visited = 1;
                }
            }
        }
    }
}

void checkCudaError(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::cerr << msg << " (error code " << err << ")!" << std::endl;
        exit(EXIT_FAILURE);
    }
}

void cudaMemcpyAsyncWithStream(unsigned int* dest, const unsigned int* src, size_t count, cudaMemcpyKind kind, cudaStream_t stream) {
    checkCudaError(cudaMemcpyAsync(dest, src, count, kind, stream), "Failed to copy data asynchronously");
}

// TOP DOWN
int main(int argc, char *argv[]) {
    string fileName = argv[1];
    unsigned int num_nodes;
    unsigned int num_edges;

    vector<vector<unsigned int>> adjacency_list = read_adjacency_list(fileName, num_nodes,num_edges);
    CSR csr;
    convert_adj_list_to_csr(adjacency_list, num_nodes, csr);

    vector<unsigned int> level(num_nodes, UINT_MAX);
    level[0] = 0;
    unsigned int new_vertex_visited = 0;
    unsigned int current_level = 0;

    unsigned int *d_row, *d_col;
    unsigned int *d_row_size, *d_col_size;
    unsigned int *d_level;
    unsigned int *d_new_vertex_visited;
    unsigned int *d_current_level;
    unsigned int *d_num_of_nodes;

    cudaStream_t stream1, stream2;
    checkCudaError(cudaStreamCreate(&stream1), "Failed to create stream1");
    checkCudaError(cudaStreamCreate(&stream2), "Failed to create stream2");

    checkCudaError(cudaMalloc(&d_row, csr.row_size * sizeof(unsigned int)), "Failed to allocate device memory for d_row");
    checkCudaError(cudaMalloc(&d_col, csr.col_size * sizeof(unsigned int)), "Failed to allocate device memory for d_col");
    checkCudaError(cudaMalloc(&d_row_size, sizeof(unsigned int)), "Failed to allocate device memory for d_row_size");
    checkCudaError(cudaMalloc(&d_col_size, sizeof(unsigned int)), "Failed to allocate device memory for d_col_size");
    checkCudaError(cudaMalloc(&d_level, num_nodes * sizeof(unsigned int)), "Failed to allocate device memory for d_level");
    checkCudaError(cudaMalloc(&d_new_vertex_visited, sizeof(unsigned int)), "Failed to allocate device memory for d_new_vertex_visited");
    checkCudaError(cudaMalloc(&d_current_level, sizeof(unsigned int)), "Failed to allocate device memory for d_current_level");
    checkCudaError(cudaMalloc(&d_num_of_nodes, sizeof(unsigned int)), "Failed to allocate device memory for d_num_of_nodes");

    cudaMemcpyAsyncWithStream(d_row, csr.row.data(), csr.row_size * sizeof(unsigned int), cudaMemcpyHostToDevice, stream1);
    cudaMemcpyAsyncWithStream(d_col, csr.col.data(), csr.col_size * sizeof(unsigned int), cudaMemcpyHostToDevice, stream2);
    cudaMemcpyAsyncWithStream(d_row_size, &csr.row_size, sizeof(unsigned int), cudaMemcpyHostToDevice, stream1);
    cudaMemcpyAsyncWithStream(d_col_size, &csr.col_size, sizeof(unsigned int), cudaMemcpyHostToDevice, stream2);
    cudaMemcpyAsyncWithStream(d_level, level.data(), num_nodes * sizeof(unsigned int), cudaMemcpyHostToDevice, stream1);
    cudaMemcpyAsyncWithStream(d_num_of_nodes, &num_nodes, sizeof(unsigned int), cudaMemcpyHostToDevice, stream2);

    checkCudaError(cudaStreamSynchronize(stream1), "Failed to synchronize stream1");
    checkCudaError(cudaStreamSynchronize(stream2), "Failed to synchronize stream2");

    unsigned int numOfThreads = 128;
    unsigned int numOfBlocks = (num_nodes + numOfThreads - 1) / numOfThreads;

    new_vertex_visited = 1;
    for (unsigned int current_level = 1; new_vertex_visited; ++current_level) {
        new_vertex_visited = 0;
        cudaMemcpyAsyncWithStream(d_current_level, &current_level, sizeof(unsigned int), cudaMemcpyHostToDevice, stream1);
        cudaMemcpyAsyncWithStream(d_new_vertex_visited, &new_vertex_visited, sizeof(unsigned int), cudaMemcpyHostToDevice, stream2);

        bfs_kernel_top_down<<<numOfBlocks, numOfThreads, 0, stream1>>>(d_row, d_col, d_row_size, d_col_size, d_level, d_new_vertex_visited, d_current_level, d_num_of_nodes);

        cudaMemcpyAsyncWithStream(&new_vertex_visited, d_new_vertex_visited, sizeof(unsigned int), cudaMemcpyDeviceToHost, stream2);

        checkCudaError(cudaStreamSynchronize(stream1), "Failed to synchronize stream1");
        checkCudaError(cudaStreamSynchronize(stream2), "Failed to synchronize stream2");
    }

    cudaMemcpyAsyncWithStream(level.data(), d_level, num_nodes * sizeof(unsigned int), cudaMemcpyDeviceToHost, stream1);
    checkCudaError(cudaStreamSynchronize(stream1), "Failed to synchronize stream1");

    ofstream output_file("output.txt");
    for (unsigned int i = 0; i < num_nodes; ++i) {
        output_file << level[i] << endl;
    }

    checkCudaError(cudaFree(d_row), "Failed to free device memory for d_row");
    checkCudaError(cudaFree(d_col), "Failed to free device memory for d_col");
    checkCudaError(cudaFree(d_row_size), "Failed to free device memory for d_row_size");
    checkCudaError(cudaFree(d_col_size), "Failed to free device memory for d_col_size");
    checkCudaError(cudaFree(d_level), "Failed to free device memory for d_level");
    checkCudaError(cudaFree(d_new_vertex_visited), "Failed to free device memory for d_new_vertex_visited");
    checkCudaError(cudaFree(d_current_level), "Failed to free device memory for d_current_level");
    checkCudaError(cudaFree(d_num_of_nodes), "Failed to free device memory for d_num_of_nodes");

    checkCudaError(cudaStreamDestroy(stream1), "Failed to destroy stream1");
    checkCudaError(cudaStreamDestroy(stream2), "Failed to destroy stream2");

    return 0;
}
