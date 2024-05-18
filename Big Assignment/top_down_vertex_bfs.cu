#include "cuda_runtime.h"
#define MAX_NODES 1000
#include <iostream>
#include <fstream>
#include <vector>
#include "utils.h"

using namespace std;


__global__
void bfs_kernel_top_down(unsigned int *d_row, unsigned int *d_col, unsigned int *d_row_size, unsigned int *d_col_size, unsigned int *d_level, unsigned int *d_new_vertex_visited, unsigned int *d_current_level ,unsigned int * num_of_nodes) {
    
    
    unsigned int vertex = blockIdx.x * blockDim.x + threadIdx.x;
    if (vertex == 0 && 0)
    {
        printf("d_row_size = %d\n", *d_row_size);
        printf("d_col_size = %d\n", *d_col_size);
        printf("d_row = ");
        for (int i = 0; i < *d_row_size; ++i) {
            printf("%d ", d_row[i]);
        }
        printf("\n");
        printf("d_col = ");
        for (int i = 0; i < *d_col_size; ++i) {
            printf("%d ", d_col[i]);
        }
        printf("\n");



    }
    if (vertex < *num_of_nodes) {
        // if vertex is in the current level
        if (d_level[vertex] == *d_current_level - 1) {
            for (unsigned int i = d_row[vertex]; i < d_row[vertex + 1]; ++i) {
                // if the neighbor is not visited
                if (d_level[d_col[i]] == UINT_MAX) {
                    d_level[d_col[i]] = *d_current_level;
                    *d_new_vertex_visited = 1;
                }
            }
        }
    }
}

// TOP DOWN
int main(int argc, char *argv[]){
    string fileName = argv[1];
    string filename = "adjacency_matrixd.txt";
    filename = fileName;
    unsigned int num_nodes;
    
    vector<vector<unsigned int>> adjacency_matrix = read_adjacency_matrix(filename, num_nodes);

    // Display the adjacency matrix
    // cout << "Number of nodes: " << num_nodes << endl;
    // cout << "Adjacency matrix:" << endl;
    // for (int i = 0; i < num_nodes; ++i) {
    //     for (int j = 0; j < num_nodes; ++j) {
    //         cout << adjacency_matrix[i][j] << " ";
    //     }
    //     cout << endl;
    // }
    CSR csr;
    convert_adj_matrix_to_csr(adjacency_matrix, num_nodes, csr);
    // Display the CSR representation
    // cout << "CSR representation:" << endl;
    // cout << "Row: ";
    // for (int i = 0; i < csr.row_size; ++i) {
    //     cout << csr.row[i] << " ";
    // }
    // cout << endl;
    // cout << "Col: ";
    // for (int i = 0; i < csr.col_size; ++i) {
    //     cout << csr.col[i] << " ";
    // }
    // cout << endl;
    // unsigned int level = 0;
    unsigned int level[MAX_NODES];
    for (int i = 0; i < num_nodes; ++i) {
        level[i] = UINT_MAX;
    }
    level[0] = 0;
    unsigned int new_vertex_visited = 0;
    unsigned int current_level = 0;
    // device variables
    unsigned int *d_row, *d_col;
    unsigned int *d_row_size, *d_col_size;
    unsigned int *d_level;
    unsigned int *d_new_vertex_visited;
    unsigned int *d_current_level;
    unsigned int *d_num_of_nodes;
    // allocate memory on the device
    cudaMalloc(&d_row, csr.row_size * sizeof(unsigned int));
    cudaMalloc(&d_col, csr.col_size * sizeof(unsigned int));
    cudaMalloc(&d_row_size, sizeof(unsigned int));
    cudaMalloc(&d_col_size, sizeof(unsigned int));
    cudaMalloc(&d_num_of_nodes, sizeof(unsigned int));
    // cudaMalloc(&d_level,sizeof(unsigned int));
    cudaMalloc(&d_level, num_nodes * sizeof(unsigned int));
    cudaMalloc(&d_new_vertex_visited,sizeof(unsigned int));
    cudaMalloc(&d_current_level,sizeof(unsigned int));
    // copy data to the device
    cudaMemcpy(d_row, csr.row.data(), csr.row_size * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col, csr.col.data(), csr.col_size * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_row_size, &csr.row_size, sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_size, &csr.col_size, sizeof(unsigned int), cudaMemcpyHostToDevice);
    // cudaMemcpy(d_level, &level, sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_level, &level, num_nodes * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_num_of_nodes, &num_nodes, sizeof(unsigned int), cudaMemcpyHostToDevice);
    // cudaMemcpy(d_new_vertex_visited, &new_vertex_visited, sizeof(unsigned int), cudaMemcpyHostToDevice);
    // cudaMemcpy(d_current_level, &current_level, sizeof(unsigned int), cudaMemcpyHostToDevice);

    // kernel call
    // number of threads per block
    unsigned int numOfThreads = 128;
    unsigned int numOfBlocks = (num_nodes + numOfThreads - 1) / numOfThreads;
    // vertex_centric_top_down<<<1, num_nodes>>>(d_row, d_col, d_row_size, d_col_size, d_level, d_new_vertex_visited, d_current_level);
    // bfs_kernel_top_down<<<numOfBlocks, numOfThreads>>>(d_row, d_col, d_row_size, d_col_size, d_level, d_new_vertex_visited, d_current_level);
    // copy data back to the host
    printf("numof Nodes = %d\n", num_nodes);
    printf("row size = %d\n", csr.row_size);
    printf("col size = %d\n", csr.col_size);
    new_vertex_visited = 1;
    for (unsigned int  current_level = 1; new_vertex_visited; ++current_level) {
        new_vertex_visited = 0;
        cudaMemcpy(d_current_level, &current_level, sizeof(unsigned int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_new_vertex_visited, &new_vertex_visited, sizeof(unsigned int), cudaMemcpyHostToDevice);
        bfs_kernel_top_down<<<numOfBlocks, numOfThreads>>>(d_row, d_col, d_row_size, d_col_size, d_level, d_new_vertex_visited, d_current_level, d_num_of_nodes);
        cudaMemcpy(&new_vertex_visited, d_new_vertex_visited, sizeof(unsigned int), cudaMemcpyDeviceToHost);
        
    }
    cudaMemcpy(level, d_level, num_nodes * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    // print the level of each vertex
    cout << "Level of each vertex:" << endl;
    for (int i = 0; i < num_nodes; ++i) {
        cout << "Vertex " << i << ": " << level[i] << endl;
    }

    // save to file
    ofstream output_file;
    output_file.open("output.txt");
    for (int i = 0; i < num_nodes; ++i) {
        output_file << level[i] << endl;
    }




    // free memory on the device
    cudaFree(d_row);
    cudaFree(d_col);
    cudaFree(d_row_size);
    cudaFree(d_col_size);
    cudaFree(d_level);
    cudaFree(d_new_vertex_visited);
    cudaFree(d_current_level);




    return 0;
}