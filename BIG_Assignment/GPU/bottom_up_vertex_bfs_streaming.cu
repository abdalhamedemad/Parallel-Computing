#include "cuda_runtime.h"
#define MAX_NODES 1000
#include <iostream>
#include <fstream>
#include <vector>
#include "utils1.h"
using namespace std;

__global__
void bfs_kernel_top_down(unsigned int *d_row, unsigned int *d_col, unsigned int *d_row_size, unsigned int *d_col_size, unsigned int *d_level, unsigned int *d_new_vertex_visited, unsigned int *d_current_level) {
    unsigned int vertex = blockIdx.x * blockDim.x + threadIdx.x;
    if (vertex < *d_row_size) {
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

__global__
void vertex_centric_bottom_up(unsigned int *d_row, unsigned int *d_col, unsigned int *d_row_size, unsigned int *d_col_size, unsigned int *d_level, unsigned int *d_new_vertex_visited, unsigned int *d_current_level, unsigned int *d_num_of_nodes) {
    unsigned int vertex = blockIdx.x * blockDim.x + threadIdx.x;
    if (vertex < *d_num_of_nodes) {
        if (d_level[vertex] == UINT_MAX) {
            for (unsigned int i = d_col[vertex]; i < d_col[vertex + 1]; ++i) {
                if (d_level[d_row[i]] == *d_current_level - 1) {
                    d_level[vertex] = *d_current_level;
                    *d_new_vertex_visited = 1;
                    break;
                }
            }
        }
    }
}

int main(int argc, char *argv[]) {
    string fileName = argv[1];
    unsigned int num_nodes;
    unsigned int num_edges;
    vector<vector<unsigned int>> adjacency_list = read_adjacency_list(fileName, num_nodes,num_edges);

    CSR csr;
    convert_adj_list_to_csr(adjacency_list, num_nodes, csr);
    CSC csc;
    convert_adj_list_to_csc(adjacency_list, num_nodes, csc);

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

    cudaMalloc(&d_row, csc.row_size * sizeof(unsigned int));
    cudaMalloc(&d_col, csc.col_size * sizeof(unsigned int));
    cudaMalloc(&d_row_size, sizeof(unsigned int));
    cudaMalloc(&d_col_size, sizeof(unsigned int));
    cudaMalloc(&d_num_of_nodes, sizeof(unsigned int));
    cudaMalloc(&d_level, num_nodes * sizeof(unsigned int));
    cudaMalloc(&d_new_vertex_visited, sizeof(unsigned int));
    cudaMalloc(&d_current_level, sizeof(unsigned int));

    cudaMemcpy(d_row, csc.row.data(), csc.row_size * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col, csc.col.data(), csc.col_size * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_row_size, &csc.row_size, sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_size, &csc.col_size, sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_level, level.data(), num_nodes * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_num_of_nodes, &num_nodes, sizeof(unsigned int), cudaMemcpyHostToDevice);

    unsigned int numOfThreads = 128;
    unsigned int numOfBlocks = (num_nodes + numOfThreads - 1) / numOfThreads;

    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);

    new_vertex_visited = 1;
    for (unsigned int current_level = 1; new_vertex_visited; ++current_level) {
        new_vertex_visited = 0;
        cudaMemcpyAsync(d_current_level, &current_level, sizeof(unsigned int), cudaMemcpyHostToDevice, stream1);
        cudaMemcpyAsync(d_new_vertex_visited, &new_vertex_visited, sizeof(unsigned int), cudaMemcpyHostToDevice, stream1);

        vertex_centric_bottom_up<<<numOfBlocks, numOfThreads, 0, stream2>>>(d_row, d_col, d_row_size, d_col_size, d_level, d_new_vertex_visited, d_current_level, d_num_of_nodes);

        cudaMemcpyAsync(&new_vertex_visited, d_new_vertex_visited, sizeof(unsigned int), cudaMemcpyDeviceToHost, stream2);
        cudaStreamSynchronize(stream2);
    }

    cudaMemcpy(level.data(), d_level, num_nodes * sizeof(unsigned int), cudaMemcpyDeviceToHost);

    ofstream output_file;
    output_file.open("output.txt");
    for (int i = 0; i < num_nodes; ++i) {
        output_file << level[i] << endl;
    }

    cudaFree(d_row);
    cudaFree(d_col);
    cudaFree(d_row_size);
    cudaFree(d_col_size);
    cudaFree(d_level);
    cudaFree(d_new_vertex_visited);
    cudaFree(d_current_level);
    cudaFree(d_num_of_nodes);

    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);

    return 0;
}
