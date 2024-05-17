#include "cuda_runtime.h"
#define MAX_NODES 1000
#include <iostream>
#include <fstream>
#include <vector>
#include "utils.h"
using namespace std;



__global__
void edge_centric(unsigned int *d_row, unsigned int *d_col, unsigned int *d_row_size, unsigned int *d_col_size, unsigned int *d_level, unsigned int *d_new_vertex_visited, unsigned int *d_current_level , unsigned int *d_num_nodes) {
    unsigned int edge = blockIdx.x * blockDim.x + threadIdx.x;
    if (edge < *d_col_size)
    {
        unsigned int vertex = d_row[edge];
        unsigned int neighbor = d_col[edge];
        if (d_level[vertex] == *d_current_level - 1 && d_level[neighbor] == UINT_MAX)
        {
            d_level[neighbor] = *d_current_level;
            *d_new_vertex_visited = 1;
        }
    }
}



// edge centric BFS
int main(){
  string filename = "adjacency_matrix2.txt";
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
    COO coo;
    convert_adj_matrix_to_coo(adjacency_matrix, num_nodes, coo);
    // Display the COO representation
    cout << "COO representation:" << endl;
    cout << "Row: ";
    for (int i = 0; i < coo.size; ++i) {
        cout << coo.row[i] << " ";
    }
    cout << endl;
    cout << "Col: ";
    for (int i = 0; i < coo.size; ++i) {
        cout << coo.col[i] << " ";
    }
    cout << endl;




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
    unsigned int *d_num_nodes;
    // allocate memory on the device
    cudaMalloc(&d_row, coo.size * sizeof(unsigned int));
    cudaMalloc(&d_col, coo.size * sizeof(unsigned int));
    cudaMalloc(&d_row_size, sizeof(unsigned int));
    cudaMalloc(&d_col_size, sizeof(unsigned int));
    // cudaMalloc(&d_level,sizeof(unsigned int));
    cudaMalloc(&d_level, num_nodes * sizeof(unsigned int));
    cudaMalloc(&d_new_vertex_visited,sizeof(unsigned int));
    cudaMalloc(&d_current_level,sizeof(unsigned int));
    cudaMalloc(&d_num_nodes, sizeof(unsigned int));
    // copy data to the device
    cudaMemcpy(d_row, coo.row.data(), coo.size * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col, coo.col.data(), coo.size * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_row_size, &coo.size, sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_size, &coo.size, sizeof(unsigned int), cudaMemcpyHostToDevice);
    // cudaMemcpy(d_level, &level, sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_level, &level, num_nodes * sizeof(unsigned int), cudaMemcpyHostToDevice);
    // cudaMemcpy(d_new_vertex_visited, &new_vertex_visited, sizeof(unsigned int), cudaMemcpyHostToDevice);
    // cudaMemcpy(d_current_level, &current_level, sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_num_nodes, &num_nodes, sizeof(unsigned int), cudaMemcpyHostToDevice);
    // kernel call
    unsigned int numOfThreads = 128;
    unsigned int numOfBlocks = (coo.size + numOfThreads - 1) / numOfThreads;
    // vertex_centric_top_down<<<1, num_nodes>>>(d_row, d_col, d_row_size, d_col_size, d_level, d_new_vertex_visited, d_current_level);
    // bfs_kernel_top_down<<<numOfBlocks, numOfThreads>>>(d_row, d_col, d_row_size, d_col_size, d_level, d_new_vertex_visited, d_current_level);
    // copy data back to the host
    // TOP DOWN
    // new_vertex_visited = 1;
    // for (unsigned int  current_level = 1; new_vertex_visited; ++current_level) {
    //     new_vertex_visited = 0;
    //     cudaMemcpy(d_current_level, &current_level, sizeof(unsigned int), cudaMemcpyHostToDevice);
    //     cudaMemcpy(d_new_vertex_visited, &new_vertex_visited, sizeof(unsigned int), cudaMemcpyHostToDevice);
    //     bfs_kernel_top_down<<<numOfBlocks, numOfThreads>>>(d_row, d_col, d_row_size, d_col_size, d_level, d_new_vertex_visited, d_current_level);
    //     cudaMemcpy(&new_vertex_visited, d_new_vertex_visited, sizeof(unsigned int), cudaMemcpyDeviceToHost);
        
    // }
    // BOTTOM UP
    new_vertex_visited = 1;
    for (unsigned int  current_level = 1; new_vertex_visited; ++current_level) {
        new_vertex_visited = 0;
        cudaMemcpy(d_current_level, &current_level, sizeof(unsigned int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_new_vertex_visited, &new_vertex_visited, sizeof(unsigned int), cudaMemcpyHostToDevice);
        edge_centric<<<numOfBlocks, numOfThreads>>>(d_row, d_col, d_row_size, d_col_size, d_level, d_new_vertex_visited, d_current_level, d_num_nodes);
        // vertex_centric_bottom_up<<<numOfBlocks, numOfThreads>>>(d_row, d_col, d_row_size, d_col_size, d_level, d_new_vertex_visited, d_current_level);
        cudaMemcpy(&new_vertex_visited, d_new_vertex_visited, sizeof(unsigned int), cudaMemcpyDeviceToHost);
        
    }
    cudaMemcpy(level, d_level, num_nodes * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    // print the level of each vertex
    cout << "Level of each vertex:" << endl;
    for (int i = 0; i < num_nodes; ++i) {
        cout << "Vertex " << i << ": " << level[i] << endl;
    }
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

