#include "cuda_runtime.h"
#define MAX_NODES 1000
#include <iostream>
#include <fstream>
#include <vector>
#include "utils.h"
using namespace std;



__global__
void vertex_centric_top_down_forntier(unsigned int *d_row, unsigned int *d_col, unsigned int *d_row_size, unsigned int *d_col_size, unsigned int *d_level, unsigned int *d_new_vertex_visited, unsigned int *d_current_level
,unsigned int * prev_forntier,  unsigned int * current_forntier, unsigned int * prev_forntier_size, unsigned int * current_forntier_index) {
    unsigned int tx = blockIdx.x * blockDim.x + threadIdx.x;
    if(tx < *prev_forntier_size){
        unsigned int vertex = prev_forntier[tx];
        for (unsigned int i = d_row[vertex]; i < d_row[vertex + 1]; ++i) {
            // here also we need atomic ops in order if threre
            // are two threads that have same neighbor can enter and add the neighbor twice
            // sol that we used atomic compare and swap to check if the neighbor already
            // visited or not atomically
            // if (d_level[d_col[i]] == UINT_MAX) {
            //     d_level[d_col[i]] = *d_current_level;
            // will compare the d_level with UINT_MAX and d_current_level atomically
            // if equal to UINT_MAX will assign d_level with d_current_level and return UINT_MAX
            if(atomicCAS(&d_level[d_col[i]],UINT_MAX,*d_current_level) == UINT_MAX){
                // here we use atomicAdd because it may occur a race condition that multiple thread
                // can edit the current_forntier idx at same time
                current_forntier[atomicAdd(current_forntier_index, 1)] = d_col[i];
                // *d_new_vertex_visited = 1;
            }
        }
    }
}
__global__
void vertex_centric_top_down_forntier_privatization(unsigned int *d_row, unsigned int *d_col, unsigned int *d_row_size, unsigned int *d_col_size, unsigned int *d_level, unsigned int *d_new_vertex_visited, unsigned int *d_current_level
,unsigned int * prev_forntier,  unsigned int * current_forntier, unsigned int * prev_forntier_size, unsigned int * current_forntier_index) {
    
    
    __shared__ unsigned int curr_forntier_shared[2048];
    __shared__ unsigned int curr_forntier_index_shared;
    // i want only one thread initialized curr_forntier_idex with zero
    if (threadIdx.x ==0)
        curr_forntier_index_shared =0;
    // to insure that non of the threads start befor intialization
    __syncthreads();
    
    unsigned int tx = blockIdx.x * blockDim.x + threadIdx.x;
    if(tx < *prev_forntier_size){
        unsigned int vertex = prev_forntier[tx];
        for (unsigned int i = d_row[vertex]; i < d_row[vertex + 1]; ++i) {
            if(atomicCAS(&d_level[d_col[i]],UINT_MAX,*d_current_level) == UINT_MAX){
                // here we use atomicAdd because it may occur a race condition that multiple thread
                // can edit the current_forntier idx at same time
                // curr_forntier_shared[atomicAdd(&curr_forntier_index_shared, 1)] = d_col[i];
                // *d_new_vertex_visited = 1;

                // if it exceed the shared memory size we will add the rest to the global memory
                if ( curr_forntier_index_shared < 2048)
                {
                    curr_forntier_shared[atomicAdd(&curr_forntier_index_shared, 1)] = d_col[i];
                    
                }else {
                current_forntier[atomicAdd(current_forntier_index, 1)] = d_col[i];
                }
            }
        }
    }
    __syncthreads();
    __shared__ unsigned int curr_forntier_start_idx;
    // copy the shared memory to the global memory
    if (threadIdx.x ==0){
    // increment the global counter by the size of the queue
        curr_forntier_start_idx =atomicAdd(current_forntier_index, curr_forntier_index_shared);
    }
    __syncthreads();
    // copy the shared memory to the global memory
    for (int i = threadIdx.x; i < curr_forntier_index_shared; i+=blockDim.x)
    {
        current_forntier[curr_forntier_start_idx + i] = curr_forntier_shared[i];
    }
}



// forntier
int main(int argc, char *argv[]){
    string fileName = argv[1];
    string filename = "adjacency_matrixd.txt";
    filename = fileName;
    unsigned int num_nodes;
    
    vector<vector<unsigned int>> adjacency_matrix = read_adjacency_matrix(filename, num_nodes);
    CSR csr;
    convert_adj_matrix_to_csr(adjacency_matrix, num_nodes, csr);

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
    unsigned int *d_prev_forntier;
    unsigned int *d_current_forntier;
    unsigned int *d_prev_forntier_size;
    unsigned int *d_current_forntier_index;
    // allocate memory on the device
    cudaMalloc(&d_row, csr.row_size * sizeof(unsigned int));
    cudaMalloc(&d_col, csr.col_size * sizeof(unsigned int));
    cudaMalloc(&d_row_size, sizeof(unsigned int));
    cudaMalloc(&d_col_size, sizeof(unsigned int));

    // cudaMalloc(&d_level,sizeof(unsigned int));
    cudaMalloc(&d_level, num_nodes * sizeof(unsigned int));
    cudaMalloc(&d_new_vertex_visited,sizeof(unsigned int));
    cudaMalloc(&d_current_level,sizeof(unsigned int));
    cudaMalloc(&d_prev_forntier, num_nodes * sizeof(unsigned int));
    cudaMalloc(&d_current_forntier, num_nodes * sizeof(unsigned int));
    cudaMalloc(&d_prev_forntier_size, sizeof(unsigned int));
    cudaMalloc(&d_current_forntier_index, sizeof(unsigned int));
    // copy data to the device
    cudaMemcpy(d_row, csr.row.data(), csr.row_size * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col, csr.col.data(), csr.col_size * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_row_size, &csr.row_size, sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_size, &csr.col_size, sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_level, &level, num_nodes * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_prev_forntier, &level[0], sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemset(d_prev_forntier_size, 1, sizeof(unsigned int));
    // kernel call
    unsigned int numOfThreads = 128;
    unsigned int numOfBlocks = (num_nodes + numOfThreads - 1) / numOfThreads;
    new_vertex_visited = 1;
    
    // TOP DOWN with forntier
    unsigned int h_prev_forntier_size =1;
    for (unsigned int  current_level = 1; (h_prev_forntier_size)> 0; ++current_level) {
        new_vertex_visited = 0;
        cudaMemcpy(d_current_level, &current_level, sizeof(unsigned int), cudaMemcpyHostToDevice);
        cudaMemset(d_current_forntier_index, 0, sizeof(unsigned int));
        unsigned int numOfBlocks = (h_prev_forntier_size + numOfThreads - 1) / numOfThreads;
        vertex_centric_top_down_forntier_privatization<<<numOfBlocks, numOfThreads>>>(d_row, d_col, d_row_size, d_col_size, d_level, d_new_vertex_visited, d_current_level, d_prev_forntier, d_current_forntier, d_prev_forntier_size, d_current_forntier_index);
        // swap
        unsigned int *temp = d_prev_forntier;
        d_prev_forntier = d_current_forntier;
        d_current_forntier = temp;
        cudaMemcpy(&h_prev_forntier_size,d_current_forntier_index,sizeof(unsigned int),cudaMemcpyDeviceToHost);
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