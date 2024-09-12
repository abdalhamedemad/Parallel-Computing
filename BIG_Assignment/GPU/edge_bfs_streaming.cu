#include "cuda_runtime.h"
#define MAX_NODES 1000
#include <iostream>
#include <fstream>
#include <vector>
#include "utils1.h"
using namespace std;

__global__
void edge_centric(unsigned int *d_row, unsigned int *d_col, unsigned int *d_row_size, unsigned int *d_col_size, unsigned int *d_level, unsigned int *d_new_vertex_visited, unsigned int *d_current_level, unsigned int *d_num_nodes) {
    unsigned int edge = blockIdx.x * blockDim.x + threadIdx.x;
    if (edge < *d_col_size) {
        unsigned int vertex = d_row[edge];
        unsigned int neighbor = d_col[edge];
        if (d_level[vertex] == *d_current_level - 1 && d_level[neighbor] == UINT_MAX) {
            d_level[neighbor] = *d_current_level;
            *d_new_vertex_visited = 1;
        }
    }
}

int main(int argc, char *argv[]) {
    string fileName = argv[1];
    unsigned int num_nodes;
    unsigned int num_edges;

    vector<vector<unsigned int>> adjacency_list = read_adjacency_list(fileName, num_nodes,num_edges);

    COO coo;
    convert_adj_list_to_coo(adjacency_list, num_nodes, coo);

    vector<unsigned int> level(num_nodes, UINT_MAX);
    level[0] = 0;
    unsigned int new_vertex_visited = 0;
    unsigned int current_level = 0;

    unsigned int *d_row, *d_col;
    unsigned int *d_row_size, *d_col_size;
    unsigned int *d_level;
    unsigned int *d_new_vertex_visited;
    unsigned int *d_current_level;
    unsigned int *d_num_nodes;

    cudaMalloc(&d_row, coo.size * sizeof(unsigned int));
    cudaMalloc(&d_col, coo.size * sizeof(unsigned int));
    cudaMalloc(&d_row_size, sizeof(unsigned int));
    cudaMalloc(&d_col_size, sizeof(unsigned int));
    cudaMalloc(&d_level, num_nodes * sizeof(unsigned int));
    cudaMalloc(&d_new_vertex_visited, sizeof(unsigned int));
    cudaMalloc(&d_current_level, sizeof(unsigned int));
    cudaMalloc(&d_num_nodes, sizeof(unsigned int));

    cudaMemcpy(d_row, coo.row.data(), coo.size * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col, coo.col.data(), coo.size * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_row_size, &coo.size, sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_size, &coo.size, sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_level, level.data(), num_nodes * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_num_nodes, &num_nodes, sizeof(unsigned int), cudaMemcpyHostToDevice);

    unsigned int numOfThreads = 128;
    unsigned int numOfBlocks = (coo.size + numOfThreads - 1) / numOfThreads;

    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);

    new_vertex_visited = 1;
    for (unsigned int current_level = 1; new_vertex_visited; ++current_level) {
        new_vertex_visited = 0;
        cudaMemcpyAsync(d_current_level, &current_level, sizeof(unsigned int), cudaMemcpyHostToDevice, stream1);
        cudaMemcpyAsync(d_new_vertex_visited, &new_vertex_visited, sizeof(unsigned int), cudaMemcpyHostToDevice, stream1);

        edge_centric<<<numOfBlocks, numOfThreads, 0, stream2>>>(d_row, d_col, d_row_size, d_col_size, d_level, d_new_vertex_visited, d_current_level, d_num_nodes);

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
    cudaFree(d_num_nodes);

    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);

    return 0;
}
