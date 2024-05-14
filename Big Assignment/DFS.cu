#include "cuda_runtime.h"
#define MAX_NODES 1000
#include <iostream>
#include <fstream>
#include <vector>

using namespace std;

vector<vector<unsigned int>> read_adjacency_matrix(string filename, unsigned int &num_nodes) {
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Error: Unable to open file " << filename << endl;
        exit(1);
    }

    file >> num_nodes;
    vector<vector<unsigned int>> adjacency_matrix(num_nodes, vector<unsigned int>(num_nodes, 0));

    for (unsigned int i = 0; i < num_nodes; ++i) {
        for (unsigned int j = 0; j < num_nodes; ++j) {
            file >> adjacency_matrix[i][j];
        }
    }

    file.close();
    return adjacency_matrix;
}

struct COO {
    unsigned int row[MAX_NODES];
    unsigned int col[MAX_NODES];
    unsigned int size;
};

__global__
void dfs_kernel(unsigned int *d_row, unsigned int *d_col, unsigned int *d_row_size, unsigned int *d_col_size, unsigned int *d_level, unsigned int *d_visited, unsigned int *d_current_vertex, unsigned int *d_stack) {
    unsigned int vertex = blockIdx.x * blockDim.x + threadIdx.x;

    if (vertex < *d_row_size && d_stack[*d_current_vertex] != UINT_MAX) {
        if (d_visited[vertex] == 0) {
            d_visited[vertex] = 1;
            d_level[vertex] = *d_current_vertex;

            unsigned int stack_idx = atomicAdd(d_current_vertex, 1);
            d_stack[stack_idx] = vertex;

            for (unsigned int i = d_row[vertex]; i < d_row[vertex + 1]; ++i) {
                unsigned int neighbor = d_col[i];
                if (d_visited[neighbor] == 0) {
                    unsigned int stack_idx = atomicAdd(d_current_vertex, 1);
                    d_stack[stack_idx] = neighbor;
                }
            }
        }
    }
}

void convert_adj_matrix_to_coo(vector<vector<unsigned int>> adjacency_matrix, unsigned int num_nodes, COO &coo) {
    coo.size = 0;

    for (int i = 0; i < num_nodes; ++i) {
        for (int j = 0; j < num_nodes; ++j) {
            if (adjacency_matrix[i][j] != 0) {
                coo.row[coo.size] = i;
                coo.col[coo.size] = j;
                ++coo.size;
            }
        }
    }
}

int main() {
    string filename = "adjacency_matrix.txt";
    unsigned int num_nodes;
    
    vector<vector<unsigned int>> adjacency_matrix = read_adjacency_matrix(filename, num_nodes);

    COO coo;
    convert_adj_matrix_to_coo(adjacency_matrix, num_nodes, coo);

    unsigned int *d_row, *d_col, *d_row_size, *d_col_size, *d_level, *d_visited, *d_stack, *d_current_vertex;
    unsigned int current_vertex = 0;

    cudaMalloc(&d_row, coo.size * sizeof(unsigned int));
    cudaMalloc(&d_col, coo.size * sizeof(unsigned int));
    cudaMalloc(&d_row_size, sizeof(unsigned int));
    cudaMalloc(&d_col_size, sizeof(unsigned int));
    cudaMalloc(&d_level, num_nodes * sizeof(unsigned int));
    cudaMalloc(&d_visited, num_nodes * sizeof(unsigned int));
    cudaMalloc(&d_stack, num_nodes * sizeof(unsigned int));
    cudaMalloc(&d_current_vertex, sizeof(unsigned int));

    cudaMemcpy(d_row, coo.row, coo.size * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col, coo.col, coo.size * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_row_size, &coo.size, sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_size, &coo.size, sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_current_vertex, &current_vertex, sizeof(unsigned int), cudaMemcpyHostToDevice);

    unsigned int numOfThreads = 128;
    unsigned int numOfBlocks = (coo.size + numOfThreads - 1) / numOfThreads;

    dfs_kernel<<<numOfBlocks, numOfThreads>>>(d_row, d_col, d_row_size, d_col_size, d_level, d_visited, d_current_vertex, d_stack);

    cudaMemcpy(adjacency_matrix.data(), d_level, num_nodes * sizeof(unsigned int), cudaMemcpyDeviceToHost);

    cout << "DFS traversal:" << endl;
    for (int i = 0; i < num_nodes; ++i) {
        cout << "Vertex " << i << ": " << adjacency_matrix[i] << endl;
    }

    cudaFree(d_row);
    cudaFree(d_col);
    cudaFree(d_row_size);
    cudaFree(d_col_size);
    cudaFree(d_level);
    cudaFree(d_visited);
    cudaFree(d_stack);
    cudaFree(d_current_vertex);

    return 0;
}
