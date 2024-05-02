import networkx as nx
import numpy as np
from matplotlib import pyplot as plt

def generate_random_graph(num_nodes, probability):
    return nx.fast_gnp_random_graph(num_nodes, probability)

def save_adjacency_matrix(graph, filename):
    num_nodes = len(graph.nodes)
    adjacency_matrix = nx.adjacency_matrix(graph)
    with open(filename, 'w') as file:
        file.write(f"{num_nodes}\n")
        np.savetxt(file, adjacency_matrix.toarray(), fmt='%d')

def main():
    num_nodes = 10  # Change this to the desired number of nodes
    probability = 0.3  # Change this to the desired probability for edge creation
    filename = "adjacency_matrix.txt"  # Change this to the desired filename
    
    # Generate random graph
    graph = generate_random_graph(num_nodes, probability)
    
    # Save adjacency matrix to file
    save_adjacency_matrix(graph, filename)
    
    print("Adjacency matrix saved to", filename)
    print("Adjacency matrix:\n", nx.adjacency_matrix(graph).toarray())
    
    # Draw the graph
    nx.draw(graph, with_labels=True)
    plt.show()

if __name__ == "__main__":
    main()
