# BFS

## Easy Way You Can use Our Notebook on Colab "Big_Assignment_Note_BOOK.ipynb"


## To RUN Code with Adjacency List (output will be in a file named "output.txt")
###  TO RUN Top-Down Approach
```bash
nvcc top_down_vertex_bfs_streaming.cu -o top_down_vertex_bfs_streaming.out

./top_down_vertex_bfs_streaming.out [input_file_name]
```
###  TO RUN Bottom-Up Approach
```bash
nvcc bottom_up_vertex_bfs_streaming.cu -o bottom_up_vertex_bfs_streaming.out

./bottom_up_vertex_bfs_streaming.out [input_file_name]
```
### Edge Centric Approach
```bash
nvcc top_down_vertex_bfs_streaming.cu -o top_down_vertex_bfs_streaming.out

./top_down_vertex_bfs_streaming.out [input_file_name]
```
### To Run Top-Down Approach with Frontier
```bash
nvcc top_down_forntier_streaming.cu -o top_down_vertex_bfs_streaming_with_frontier.out

./top_down_vertex_bfs_streaming_with_frontier.out [input_file_name]
```

### To Run Top-Down Approach with Frontier and Privatization
```bash
nvcc top_down_forntier_privetization_adj.cu -o top_down_vertex_bfs_streaming_with_frontier_priv.out

./top_down_vertex_bfs_streaming_with_frontier_priv.out [input_file_name]
```

## To RUN Code with Adjacency Matrix U can use any of the other files and instead of adding adjacency list add adjacency matrix

## To Compare output of two files (add name inside compare.py file)
```bash
python compare.py
```
