from collections import deque

def read_adjacency_matrix(filename):
    with open(filename, 'r') as file:
        num_nodes = int(file.readline())
        adjacency_matrix = [[int(x) for x in line.split()] for line in file]
    return adjacency_matrix, num_nodes

def bfs(adj_matrix, start):
    num_nodes = len(adj_matrix)
    visited = [False] * num_nodes
    level = [None] * num_nodes

    queue = deque()
    queue.append(start)
    visited[start] = True
    level[start] = 0

    level_dict = {}

    while queue:
        current_node = queue.popleft()
        level_dict[current_node] = level[current_node]

        neighbors = [neighbor for neighbor in range(num_nodes) if adj_matrix[current_node][neighbor] == 1 and not visited[neighbor]]
        for neighbor in neighbors:
            queue.append(neighbor)
            visited[neighbor] = True
            level[neighbor] = level[current_node] + 1

    sorted_levels = sorted(level_dict.items(), key=lambda x: x[0])  # Sort levels based on node number

    for node, level in sorted_levels:
        print("Node:", node, "Level:", level)
    # save the levels to a file
    with open("bfs_levels.txt", "w") as file:
        for node, level in sorted_levels:
            file.write(f"{node} {level}\n")

def main():
    filename = "adjacency_matrix.txt"  # Change this to the filename of your adjacency matrix

    adjacency_matrix, num_nodes = read_adjacency_matrix(filename)
    
    start_node = 0  # Starting node for BFS
    print("Starting BFS from node", start_node)
    bfs(adjacency_matrix, start_node)

if __name__ == "__main__":
    main()
