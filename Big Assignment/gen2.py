import numpy as np

def generate_deep_adjacency_matrix(levels):
    """
    Generate an adjacency matrix for a deep hierarchical structure (binary tree).
    
    Args:
    levels (int): Number of levels in the hierarchical structure.
    
    Returns:
    numpy.ndarray: The adjacency matrix representing the structure.
    """
    if levels < 1:
        raise ValueError("Number of levels must be at least 1")
    
    # Calculate the total number of nodes in a binary tree with given levels
    num_nodes = 2**levels - 1
    
    # Initialize the adjacency matrix with zeros
    adjacency_matrix = np.zeros((num_nodes, num_nodes), dtype=int)
    
    # Populate the adjacency matrix to represent a binary tree
    for i in range(num_nodes):
        left_child = 2 * i + 1
        right_child = 2 * i + 2
        
        if left_child < num_nodes:
            adjacency_matrix[i, left_child] = 1
        
        if right_child < num_nodes:
            adjacency_matrix[i, right_child] = 1
    
    return adjacency_matrix

def save_adjacency_matrix_to_file(adjacency_matrix, filename):
    """
    Save the adjacency matrix to a file with the number of nodes.
    
    Args:
    adjacency_matrix (numpy.ndarray): The adjacency matrix to save.
    filename (str): The name of the file to save the matrix to.
    """
    num_nodes = adjacency_matrix.shape[0]
    
    with open(filename, 'w') as f:
        f.write(f"{num_nodes}\n")
        np.savetxt(f, adjacency_matrix, fmt='%d')

# Example usage:
levels = 9  # Number of levels in the tree
adj_matrix = generate_deep_adjacency_matrix(levels)

# Save the adjacency matrix to a .txt file
filename = 'adjacency_matrixd.txt'
save_adjacency_matrix_to_file(adj_matrix, filename)

print(f"Adjacency matrix with {adj_matrix.shape[0]} nodes saved to {filename}")
