#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>

using namespace std;

vector<vector<unsigned int>> read_adjacency_list(string filename, unsigned int &num_nodes, unsigned int &num_edges) {
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Error: Unable to open file " << filename << endl;
        exit(1);
    }

    file >> num_nodes >> num_edges;
    vector<vector<unsigned int>> adjacency_list(num_nodes);

    string line;
    getline(file, line);  // Skip the rest of the first line
    for (unsigned int node = 0; node < num_nodes; ++node) {
        getline(file, line);
        stringstream ss(line);
        unsigned int neighbor;
        ss >> neighbor;  // Read the first node (should be node itself)
        adjacency_list[node].push_back(neighbor);
        while (ss >> neighbor) {
            adjacency_list[node].push_back(neighbor);
        }
    }

    file.close();
    return adjacency_list;
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

void convert_adj_list_to_csr(const vector<vector<unsigned int>> &adjacency_list, unsigned int num_nodes, CSR &csr) {
    csr.row.resize(num_nodes + 1);
    csr.col.clear();
    csr.row[0] = 0;

    for (unsigned int i = 0; i < num_nodes; ++i) {
        for (unsigned int j : adjacency_list[i]) {
            csr.col.push_back(j);
        }
        csr.row[i + 1] = csr.col.size();
    }
    csr.row_size = num_nodes + 1;
    csr.col_size = csr.col.size();
}

void convert_adj_list_to_csc(const vector<vector<unsigned int>> &adjacency_list, unsigned int num_nodes, CSC &csc) {
    csc.col.resize(num_nodes + 1);
    csc.row.clear();
    csc.col[0] = 0;

    vector<unsigned int> count(num_nodes, 0);
    for (unsigned int i = 0; i < num_nodes; ++i) {
        for (unsigned int j : adjacency_list[i]) {
            count[j]++;
        }
    }

    for (unsigned int i = 1; i <= num_nodes; ++i) {
        csc.col[i] = csc.col[i - 1] + count[i - 1];
    }

    csc.row.resize(csc.col[num_nodes]);
    vector<unsigned int> index(num_nodes, 0);

    for (unsigned int i = 0; i < num_nodes; ++i) {
        for (unsigned int j : adjacency_list[i]) {
            unsigned int col_index = csc.col[j] + index[j];
            csc.row[col_index] = i;
            index[j]++;
        }
    }

    csc.col_size = num_nodes + 1;
    csc.row_size = csc.row.size();
}

void convert_adj_list_to_coo(const vector<vector<unsigned int>> &adjacency_list, unsigned int num_nodes, COO &coo) {
    coo.row.clear();
    coo.col.clear();
    coo.size = 0;

    for (unsigned int i = 0; i < num_nodes; ++i) {
        for (unsigned int j : adjacency_list[i]) {
            coo.row.push_back(i);
            coo.col.push_back(j);
            ++coo.size;
        }
    }
}