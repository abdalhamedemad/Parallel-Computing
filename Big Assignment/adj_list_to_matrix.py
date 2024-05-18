def read_adj_list_from_file(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()[1:]  # Skip the first line
        adj_list = [list(map(int, line.split())) for line in lines]
    return adj_list

def convert_adj_list_to_matrix(adj_list):
    num_nodes = len(adj_list)
    adj_matrix = [[0] * num_nodes for _ in range(num_nodes)]
    for i, neighbors in enumerate(adj_list):
        for neighbor in neighbors:
            adj_matrix[i][neighbor] = 1
    return adj_matrix, num_nodes

def write_matrix_to_file(filename, adj_matrix, num_nodes):
    with open(filename, 'w') as file:
        file.write(f"Number of nodes: {num_nodes}\n")
        for row in adj_matrix:
            file.write(" ".join(map(str, row)) + "\n")

# File paths
input_file = 'list.txt'
output_file = 'matrix.txt'

# Read, convert, and write
adj_list = read_adj_list_from_file(input_file)
adj_matrix, num_nodes = convert_adj_list_to_matrix(adj_list)
write_matrix_to_file(output_file, adj_matrix, num_nodes)
