#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include "utils1.h"
using namespace std;

// Kernel definitions remain unchanged
__global__
void vertex_centric_top_down_forntier(unsigned int *d_row, unsigned int *d_col, unsigned int *d_row_size, unsigned int *d_col_size, unsigned int *d_level, unsigned int *d_new_vertex_visited, unsigned int *d_current_level,
                                      unsigned int *prev_forntier, unsigned int *current_forntier, unsigned int *prev_forntier_size, unsigned int *current_forntier_index) {
    unsigned int tx = blockIdx.x * blockDim.x + threadIdx.x;
    if (tx < *prev_forntier_size) {
        unsigned int vertex = prev_forntier[tx];
        for (unsigned int i = d_row[vertex]; i < d_row[vertex + 1]; ++i) {
            if (atomicCAS(&d_level[d_col[i]], UINT_MAX, *d_current_level) == UINT_MAX) {
                current_forntier[atomicAdd(current_forntier_index, 1)] = d_col[i];
            }
        }
    }
}

// Main function
int main(int argc, char *argv[]) {
    string fileName = argv[1];
    string filename = "adjacency_matrixd.txt";
    filename = fileName;
    unsigned int num_nodes;
    unsigned int num_edges;

    vector<vector<unsigned int>> adjacency_list = read_adjacency_list(filename, num_nodes,num_edges);
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
    unsigned int *d_prev_forntier;
    unsigned int *d_current_forntier;
    unsigned int *d_prev_forntier_size;
    unsigned int *d_current_forntier_index;

    cudaMalloc(&d_row, csr.row_size * sizeof(unsigned int));
    cudaMalloc(&d_col, csr.col_size * sizeof(unsigned int));
    cudaMalloc(&d_row_size, sizeof(unsigned int));
    cudaMalloc(&d_col_size, sizeof(unsigned int));
    cudaMalloc(&d_level, num_nodes * sizeof(unsigned int));
    cudaMalloc(&d_new_vertex_visited, sizeof(unsigned int));
    cudaMalloc(&d_current_level, sizeof(unsigned int));
    cudaMalloc(&d_prev_forntier, num_nodes * sizeof(unsigned int));
    cudaMalloc(&d_current_forntier, num_nodes * sizeof(unsigned int));
    cudaMalloc(&d_prev_forntier_size, sizeof(unsigned int));
    cudaMalloc(&d_current_forntier_index, sizeof(unsigned int));

    cudaMemcpy(d_row, csr.row.data(), csr.row_size * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col, csr.col.data(), csr.col_size * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_row_size, &csr.row_size, sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_size, &csr.col_size, sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_level, level.data(), num_nodes * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_prev_forntier, &level[0], sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemset(d_prev_forntier_size, 1, sizeof(unsigned int));

    // Create CUDA streams
    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);

    auto start = chrono::system_clock::now();
    unsigned int numOfThreads = 128;
    unsigned int h_prev_forntier_size = 1;

    for (unsigned int current_level = 1; h_prev_forntier_size > 0; ++current_level) {
        new_vertex_visited = 0;

        cudaMemcpyAsync(d_current_level, &current_level, sizeof(unsigned int), cudaMemcpyHostToDevice, stream1);
        cudaMemsetAsync(d_current_forntier_index, 0, sizeof(unsigned int), stream2);

        unsigned int numOfBlocks = (h_prev_forntier_size + numOfThreads - 1) / numOfThreads;

        vertex_centric_top_down_forntier<<<numOfBlocks, numOfThreads, 0, stream1>>>(d_row, d_col, d_row_size, d_col_size, d_level, d_new_vertex_visited, d_current_level, d_prev_forntier, d_current_forntier, d_prev_forntier_size, d_current_forntier_index);

        cudaMemcpyAsync(&h_prev_forntier_size, d_current_forntier_index, sizeof(unsigned int), cudaMemcpyDeviceToHost, stream1);
        cudaStreamSynchronize(stream1);

        unsigned int *temp = d_prev_forntier;
        d_prev_forntier = d_current_forntier;
        d_current_forntier = temp;
    }

    auto end = chrono::system_clock::now();
    chrono::duration<double> elapsed_seconds = end - start;
    cout << "Elapsed time: " << elapsed_seconds.count() << "s\n";

    cudaMemcpy(level.data(), d_level, num_nodes * sizeof(unsigned int), cudaMemcpyDeviceToHost);

    // Save to file
    ofstream output_file;
    output_file.open("output.txt");
    for (int i = 0; i < num_nodes; ++i) {
        output_file << level[i] << endl;
    }

    // Free memory on the device
    cudaFree(d_row);
    cudaFree(d_col);
    cudaFree(d_row_size);
    cudaFree(d_col_size);
    cudaFree(d_level);
    cudaFree(d_new_vertex_visited);
    cudaFree(d_current_level);
    cudaFree(d_prev_forntier);
    cudaFree(d_current_forntier);
    cudaFree(d_prev_forntier_size);
    cudaFree(d_current_forntier_index);

    // Destroy CUDA streams
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);

    return 0;
}
