#include <iostream>
#include <fstream>
#include <vector>
#include <queue>
#include "utils.h" 
// include time
#include <chrono>
#include <ctime>

using namespace std;

int main(int argc, char *argv[]) {
    // start time
    string fileName = argv[1];
    string filename = "adjacency_matrixd.txt";
    filename = fileName;
    unsigned int num_nodes;

    vector<vector<unsigned int>> adjacency_matrix = read_adjacency_matrix(filename, num_nodes);

    CSR csr;
    convert_adj_matrix_to_csr(adjacency_matrix, num_nodes, csr);

    auto start = chrono::system_clock::now();
    vector<unsigned int> level(num_nodes, UINT_MAX);
    level[0] = 0; // Set the source vertex level to 0

    queue<unsigned int> q;
    q.push(0); // Enqueue the source vertex

    while (!q.empty()) {
        unsigned int u = q.front();
        q.pop();

        // Traverse all the neighbors of vertex u
        for (unsigned int i = csr.row[u]; i < csr.row[u + 1]; ++i) {
            unsigned int v = csr.col[i];
            if (level[v] == UINT_MAX) {
                level[v] = level[u] + 1;
                q.push(v);
            }
        }
    }
    auto end = chrono::system_clock::now();
    chrono::duration<double> elapsed_seconds = end - start;
    // Print the elapsed time
    cout << "Elapsed time: " << elapsed_seconds.count() << "s" << endl;
    // Print the level of each vertex
    cout << "Level of each vertex:" << endl;
    for (int i = 0; i < num_nodes; ++i) {
        cout << "Vertex " << i << ": " << level[i] << endl;
    }

    // Save the levels to a file
    ofstream output_file;
    output_file.open("output.txt");
    for (int i = 0; i < num_nodes; ++i) {
        output_file << level[i] << endl;
    }
    output_file.close();

    return 0;
}