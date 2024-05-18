#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>

using namespace std;


vector<vector<unsigned int>> read_adj_list_from_file(string& filename) {
    ifstream file(filename);
    string line;
    vector<vector<unsigned int>> adjList;

    // Skip the first line
    std::getline(file, line);

    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::vector<unsigned int> neighbors;
        unsigned int neighbor;

        while (iss >> neighbor) {
            neighbors.push_back(neighbor);
        }

        adjList.push_back(neighbors);
    }

    return adjList;
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

int main(int argc, char *argv[]) {
    if (argc < 2) {
        cerr << "Usage: " << argv[0] << " <filename>" << endl;
        return 1;
    }

    string filename = argv[1];
    unsigned int num_nodes;
    unsigned int num_edges;

    vector<vector<unsigned int>> adjacency_list = read_adj_list_from_file(filename);
    num_nodes= adjacency_list.size();
    cout << "Number of nodes: " << num_nodes << endl;

    // Output the adjacency list to verify the result
    // for (unsigned int i = 0; i < adjacency_list.size(); ++i) {
    //     cout << "Node " << i << ":";
    //     for (unsigned int j = 0; j < adjacency_list[i].size(); ++j) {
    //         cout << " " << adjacency_list[i][j];
    //     }
    //     cout << endl;
    // }
    CSR csr;
    convert_adj_list_to_csr(adjacency_list, num_nodes, csr);

    cout << "CSR Representation:" << endl;
    cout << "Row size: " << csr.row_size << endl;
    cout << "Col size: " << csr.col_size << endl;

    CSC csc;
    convert_adj_list_to_csc(adjacency_list, num_nodes, csc);

    cout << "CSC Representation:" << endl;
    cout << "Col size: " << csc.col_size << endl;
    cout << "Row size: " << csc.row_size << endl;

    COO coo;
    convert_adj_list_to_coo(adjacency_list, num_nodes, coo);

    cout << "COO Representation:" << endl;
    cout << "Size: " << coo.size << endl;

    // Output or further processing of csr, csc, and coo structures

    return 0;
}
