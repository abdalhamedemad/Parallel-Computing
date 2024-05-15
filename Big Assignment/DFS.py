from collections import deque
def read_adjacency_matrix(filename):
    with open(filename, 'r') as file:
        num_nodes = int(file.readline())
        adjacency_matrix = [[int(x) for x in line.split()] for line in file]
    return adjacency_matrix, num_nodes

def dfs(adj_matrix, start):
    num_nodes = len(adj_matrix)
    visited = [False] * num_nodes
    traversal_order = []

    def dfs_recursive(node):
        visited[node] = True
        traversal_order.append(node)

        for neighbor in range(num_nodes):
            if adj_matrix[node][neighbor] == 1 and not visited[neighbor]:
                dfs_recursive(neighbor)

    dfs_recursive(start)

    # save the traversal order to a file
    with open("dfs_traversal_order.txt", "w") as file:
        for node in traversal_order:
            file.write(f"{node}\n")

def main():
    filename = "adjacency_matrix.txt"  # Change this to the filename of your adjacency matrix

    adjacency_matrix, num_nodes = read_adjacency_matrix(filename)
    
    start_node = 0  # Starting node for DFS
    print("Starting DFS from node", start_node)
    dfs(adjacency_matrix, start_node)

if __name__ == "__main__":
    main()
