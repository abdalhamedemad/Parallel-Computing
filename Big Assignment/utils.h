#include <iostream>
#include <fstream>
#include <vector>
#define MAX_NODES 1000

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

struct CSC {
    vector<unsigned int> col;
    vector<unsigned int> row;
    unsigned int col_size;
    unsigned int row_size;
};

struct COO {
    vector<unsigned int> row;
    vector<unsigned int> col;
    unsigned int size;
};
struct CSR {
    vector<unsigned int> row;
    vector<unsigned int> col;
    unsigned int row_size;
    unsigned int col_size;
};


void convert_adj_matrix_to_csr(vector<vector<unsigned int>> &adjacency_matrix, unsigned int num_nodes, CSR &csr) {
    csr.row.resize(num_nodes + 1);
    csr.col.clear();
    csr.row[0] = 0;
    csr.row_size = 1;
    csr.col_size = 0;

    printf("num_nodes = %d\n", num_nodes);

    for (unsigned int i = 0; i < num_nodes; ++i) {
        for (unsigned int j = 0; j < num_nodes; ++j) {
            if (adjacency_matrix[i][j] != 0) {
                csr.col.push_back(j);
            }
        }
        csr.row[i + 1] = csr.col.size();
    }
    csr.row_size = num_nodes + 1;
    csr.col_size = csr.col.size();

    printf("csr.row_size = %d\n", csr.row_size);
}

void convert_adj_matrix_to_csc(vector<vector<unsigned int>> &adjacency_matrix, unsigned int num_nodes, CSC &csc) {
    csc.col.resize(num_nodes + 1);
    csc.row.clear();
    csc.col[0] = 0;
    csc.col_size = 1;
    csc.row_size = 0;

    for (unsigned int i = 0; i < num_nodes; ++i) {
        for (unsigned int j = 0; j < num_nodes; ++j) {
            if (adjacency_matrix[j][i] != 0) {
                csc.row.push_back(j);
            }
        }
        csc.col[i + 1] = csc.row.size();
    }
    csc.col_size = num_nodes + 1;
    csc.row_size = csc.row.size();
}

void convert_adj_matrix_to_coo(vector<vector<unsigned int>> &adjacency_matrix, unsigned int num_nodes, COO &coo) {
    coo.row.clear();
    coo.col.clear();
    coo.size = 0;

    for (unsigned int i = 0; i < num_nodes; ++i) {
        for (unsigned int j = 0; j < num_nodes; ++j) {
            if (adjacency_matrix[i][j] != 0) {
                coo.row.push_back(i);
                coo.col.push_back(j);
                ++coo.size;
            }
        }
    }
}


