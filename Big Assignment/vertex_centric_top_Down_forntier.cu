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
struct CSR {
    unsigned int row[MAX_NODES];
    unsigned int col[MAX_NODES];
    unsigned int row_size;
    unsigned int col_size;
};
struct CSC {
    unsigned int col[MAX_NODES];
    unsigned int row[MAX_NODES];
    unsigned int col_size;
    unsigned int row_size;
};
__global__
void bfs_kernel_top_down(unsigned int *d_row, unsigned int *d_col, unsigned int *d_row_size, unsigned int *d_col_size, unsigned int *d_level, unsigned int *d_new_vertex_visited, unsigned int *d_current_level) {
    
    
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
void vertex_centric_bottom_up(unsigned int *d_row, unsigned int *d_col, unsigned int *d_row_size, unsigned int *d_col_size, unsigned int *d_level, unsigned int *d_new_vertex_visited, unsigned int *d_current_level) {
    unsigned int vertex = blockIdx.x * blockDim.x + threadIdx.x;
    if (vertex < *d_col_size)
    {
        if (d_level[vertex] == UINT_MAX)
        {
            for (unsigned int i = d_col[vertex]; i < d_col[vertex + 1]; ++i)
            {
                if (d_level[d_row[i]] == *d_current_level -1)
                {
                    d_level[vertex] = *d_current_level;
                    *d_new_vertex_visited = 1;
                    break;
                }
            }
        }
    }
}

__global__
void edge_centric(unsigned int *d_row, unsigned int *d_col, unsigned int *d_row_size, unsigned int *d_col_size, unsigned int *d_level, unsigned int *d_new_vertex_visited, unsigned int *d_current_level) {
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
void convert_adj_matrix_to_csr(vector<vector<unsigned int>> adjacency_matrix, unsigned int num_nodes, CSR &csr) {
    csr.row[0] = 0;
    csr.row_size = 1;
    csr.col_size = 0;

    for (int i = 0; i < num_nodes; ++i) {
        for (int j = 0; j < num_nodes; ++j) {
            if (adjacency_matrix[i][j] != 0) {
                csr.col[csr.col_size++] = j;
            }
        }
        csr.row[csr.row_size] = csr.col_size;
        ++csr.row_size;
    }
}

void convert_adj_matrix_to_csc(vector<vector<unsigned int>> adjacency_matrix, unsigned int num_nodes, CSC &csc) {
    csc.col[0] = 0;
    csc.col_size = 1;
    csc.row_size = 0;

    for (int i = 0; i < num_nodes; ++i) {
        for (int j = 0; j < num_nodes; ++j) {
            if (adjacency_matrix[j][i] != 0) {
                csc.row[csc.row_size++] = j;
            }
        }
        csc.col[csc.col_size] = csc.row_size;
        ++csc.col_size;
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


// edge centric BFS
int main2(){
  string filename = "adjacency_matrix.txt";
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
    // allocate memory on the device
    cudaMalloc(&d_row, coo.size * sizeof(unsigned int));
    cudaMalloc(&d_col, coo.size * sizeof(unsigned int));
    cudaMalloc(&d_row_size, sizeof(unsigned int));
    cudaMalloc(&d_col_size, sizeof(unsigned int));
    // cudaMalloc(&d_level,sizeof(unsigned int));
    cudaMalloc(&d_level, num_nodes * sizeof(unsigned int));
    cudaMalloc(&d_new_vertex_visited,sizeof(unsigned int));
    cudaMalloc(&d_current_level,sizeof(unsigned int));
    // copy data to the device
    cudaMemcpy(d_row, coo.row, coo.size * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col, coo.col, coo.size * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_row_size, &coo.size, sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_size, &coo.size, sizeof(unsigned int), cudaMemcpyHostToDevice);
    // cudaMemcpy(d_level, &level, sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_level, &level, num_nodes * sizeof(unsigned int), cudaMemcpyHostToDevice);
    // cudaMemcpy(d_new_vertex_visited, &new_vertex_visited, sizeof(unsigned int), cudaMemcpyHostToDevice);
    // cudaMemcpy(d_current_level, &current_level, sizeof(unsigned int), cudaMemcpyHostToDevice);

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
        edge_centric<<<numOfBlocks, numOfThreads>>>(d_row, d_col, d_row_size, d_col_size, d_level, d_new_vertex_visited, d_current_level);
        // vertex_centric_bottom_up<<<numOfBlocks, numOfThreads>>>(d_row, d_col, d_row_size, d_col_size, d_level, d_new_vertex_visited, d_current_level);
        cudaMemcpy(&new_vertex_visited, d_new_vertex_visited, sizeof(unsigned int), cudaMemcpyDeviceToHost);
        
    }
    cudaMemcpy(level, d_level, num_nodes * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    // print the level of each vertex
    cout << "Level of each vertex:" << endl;
    for (int i = 0; i < num_nodes; ++i) {
        cout << "Vertex " << i << ": " << level[i] << endl;
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


// Direction optimized
// not completed
int main5(){
  string filename = "adjacency_matrix.txt";
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
    CSC csc;
    convert_adj_matrix_to_csc(adjacency_matrix, num_nodes, csc);
    // Display the CSC  representation
    cout << "CSC representation:" << endl;
    cout << "Col: ";
    for (int i = 0; i < csc.col_size; ++i) {
        cout << csc.col[i] << " ";
    }
    cout << endl;
    cout << "Row: ";
    for (int i = 0; i < csc.row_size; ++i) {
        cout << csc.row[i] << " ";
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
    // allocate memory on the device
    cudaMalloc(&d_row, csc.row_size * sizeof(unsigned int));
    cudaMalloc(&d_col, csc.col_size * sizeof(unsigned int));
    cudaMalloc(&d_row_size, sizeof(unsigned int));
    cudaMalloc(&d_col_size, sizeof(unsigned int));
    // cudaMalloc(&d_level,sizeof(unsigned int));
    cudaMalloc(&d_level, num_nodes * sizeof(unsigned int));
    cudaMalloc(&d_new_vertex_visited,sizeof(unsigned int));
    cudaMalloc(&d_current_level,sizeof(unsigned int));
    // copy data to the device
    cudaMemcpy(d_row, csc.row, csc.row_size * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col, csc.col, csc.col_size * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_row_size, &csc.row_size, sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_size, &csc.col_size, sizeof(unsigned int), cudaMemcpyHostToDevice);
    // cudaMemcpy(d_level, &level, sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_level, &level, num_nodes * sizeof(unsigned int), cudaMemcpyHostToDevice);
    // cudaMemcpy(d_new_vertex_visited, &new_vertex_visited, sizeof(unsigned int), cudaMemcpyHostToDevice);
    // cudaMemcpy(d_current_level, &current_level, sizeof(unsigned int), cudaMemcpyHostToDevice);

    // kernel call
    unsigned int numOfThreads = 128;
    unsigned int numOfBlocks = (num_nodes + numOfThreads - 1) / numOfThreads;
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
        if (current_level == 1)
        {   
            bfs_kernel_top_down<<<numOfBlocks, numOfThreads>>>(d_row, d_col, d_row_size, d_col_size, d_level, d_new_vertex_visited, d_current_level);
        }
        else
        {
            vertex_centric_bottom_up<<<numOfBlocks, numOfThreads>>>(d_row, d_col, d_row_size, d_col_size, d_level, d_new_vertex_visited, d_current_level);
        }
        // vertex_centric_bottom_up<<<numOfBlocks, numOfThreads>>>(d_row, d_col, d_row_size, d_col_size, d_level, d_new_vertex_visited, d_current_level);
        cudaMemcpy(&new_vertex_visited, d_new_vertex_visited, sizeof(unsigned int), cudaMemcpyDeviceToHost);
        
    }
    cudaMemcpy(level, d_level, num_nodes * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    // print the level of each vertex
    cout << "Level of each vertex:" << endl;
    for (int i = 0; i < num_nodes; ++i) {
        cout << "Vertex " << i << ": " << level[i] << endl;
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



// Bottom up

int main3(){
  string filename = "adjacency_matrix.txt";
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
    CSC csc;
    convert_adj_matrix_to_csc(adjacency_matrix, num_nodes, csc);
    // Display the CSC  representation
    cout << "CSC representation:" << endl;
    cout << "Col: ";
    for (int i = 0; i < csc.col_size; ++i) {
        cout << csc.col[i] << " ";
    }
    cout << endl;
    cout << "Row: ";
    for (int i = 0; i < csc.row_size; ++i) {
        cout << csc.row[i] << " ";
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
    // allocate memory on the device
    cudaMalloc(&d_row, csc.row_size * sizeof(unsigned int));
    cudaMalloc(&d_col, csc.col_size * sizeof(unsigned int));
    cudaMalloc(&d_row_size, sizeof(unsigned int));
    cudaMalloc(&d_col_size, sizeof(unsigned int));
    // cudaMalloc(&d_level,sizeof(unsigned int));
    cudaMalloc(&d_level, num_nodes * sizeof(unsigned int));
    cudaMalloc(&d_new_vertex_visited,sizeof(unsigned int));
    cudaMalloc(&d_current_level,sizeof(unsigned int));
    // copy data to the device
    cudaMemcpy(d_row, csc.row, csc.row_size * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col, csc.col, csc.col_size * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_row_size, &csc.row_size, sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_size, &csc.col_size, sizeof(unsigned int), cudaMemcpyHostToDevice);
    // cudaMemcpy(d_level, &level, sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_level, &level, num_nodes * sizeof(unsigned int), cudaMemcpyHostToDevice);
    // cudaMemcpy(d_new_vertex_visited, &new_vertex_visited, sizeof(unsigned int), cudaMemcpyHostToDevice);
    // cudaMemcpy(d_current_level, &current_level, sizeof(unsigned int), cudaMemcpyHostToDevice);

    // kernel call
    unsigned int numOfThreads = 128;
    unsigned int numOfBlocks = (num_nodes + numOfThreads - 1) / numOfThreads;
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
        vertex_centric_bottom_up<<<numOfBlocks, numOfThreads>>>(d_row, d_col, d_row_size, d_col_size, d_level, d_new_vertex_visited, d_current_level);
        cudaMemcpy(&new_vertex_visited, d_new_vertex_visited, sizeof(unsigned int), cudaMemcpyDeviceToHost);
        
    }
    cudaMemcpy(level, d_level, num_nodes * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    // print the level of each vertex
    cout << "Level of each vertex:" << endl;
    for (int i = 0; i < num_nodes; ++i) {
        cout << "Vertex " << i << ": " << level[i] << endl;
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
// TOP DOWN
int main(){
  string filename = "adjacency_matrix.txt";
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
    CSC csc;
    convert_adj_matrix_to_csc(adjacency_matrix, num_nodes, csc);
    // Display the CSC  representation
    cout << "CSC representation:" << endl;
    cout << "Col: ";
    for (int i = 0; i < csc.col_size; ++i) {
        cout << csc.col[i] << " ";
    }
    cout << endl;
    cout << "Row: ";
    for (int i = 0; i < csc.row_size; ++i) {
        cout << csc.row[i] << " ";
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
    cudaMemcpy(d_row, csr.row, csr.row_size * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col, csr.col, csr.col_size * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_row_size, &csr.row_size, sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_size, &csr.col_size, sizeof(unsigned int), cudaMemcpyHostToDevice);
    // cudaMemcpy(d_level, &level, sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_level, &level, num_nodes * sizeof(unsigned int), cudaMemcpyHostToDevice);
    // cudaMemcpy(d_new_vertex_visited, &new_vertex_visited, sizeof(unsigned int), cudaMemcpyHostToDevice);
    // cudaMemcpy(d_current_level, &current_level, sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_prev_forntier, &level[0], sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemset(d_prev_forntier_size, 1, sizeof(unsigned int));
    // kernel call
    unsigned int numOfThreads = 128;
    unsigned int numOfBlocks = (num_nodes + numOfThreads - 1) / numOfThreads;
    // vertex_centric_top_down<<<1, num_nodes>>>(d_row, d_col, d_row_size, d_col_size, d_level, d_new_vertex_visited, d_current_level);
    // bfs_kernel_top_down<<<numOfBlocks, numOfThreads>>>(d_row, d_col, d_row_size, d_col_size, d_level, d_new_vertex_visited, d_current_level);
    // copy data back to the host
    new_vertex_visited = 1;
    // for (unsigned int  current_level = 1; new_vertex_visited; ++current_level) {
    //     new_vertex_visited = 0;
    //     cudaMemcpy(d_current_level, &current_level, sizeof(unsigned int), cudaMemcpyHostToDevice);
    //     cudaMemcpy(d_new_vertex_visited, &new_vertex_visited, sizeof(unsigned int), cudaMemcpyHostToDevice);
    //     bfs_kernel_top_down<<<numOfBlocks, numOfThreads>>>(d_row, d_col, d_row_size, d_col_size, d_level, d_new_vertex_visited, d_current_level);
    //     cudaMemcpy(&new_vertex_visited, d_new_vertex_visited, sizeof(unsigned int), cudaMemcpyDeviceToHost);
        
    // }
    // TOP DOWN with forntier
    unsigned int h_prev_forntier_size =1;
    for (unsigned int  current_level = 1; (h_prev_forntier_size)> 0; ++current_level) {
        new_vertex_visited = 0;
        cudaMemcpy(d_current_level, &current_level, sizeof(unsigned int), cudaMemcpyHostToDevice);
        cudaMemset(d_current_forntier_index, 0, sizeof(unsigned int));
        unsigned int numOfBlocks = (h_prev_forntier_size + numOfThreads - 1) / numOfThreads;
        vertex_centric_top_down_forntier<<<numOfBlocks, numOfThreads>>>(d_row, d_col, d_row_size, d_col_size, d_level, d_new_vertex_visited, d_current_level, d_prev_forntier, d_current_forntier, d_prev_forntier_size, d_current_forntier_index);
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