import numpy as np
import sys
from itertools import combinations
import tqdm


# Graph class
class Graph:
    # Store the graph as set of adjacency lists of each nodes
    # Assume undirected graph
    # For example, if we have node 1, node 2 and edge (1, 2), (1, 3),
    # adj_list = {1: [2, 3], 2: [1], 3: [1]}
    def __init__(self):
        self.adj_list = {}

    # Add an edge to the graph (i.e., update adj_list)
    def add_edge(self, u, v):
        if u not in self.adj_list:
            self.adj_list[u] = []
        self.adj_list[u].append(v)
        self.adj_list[u].sort()

        if v not in self.adj_list:
            self.adj_list[v] = []
        self.adj_list[v].append(u)
        self.adj_list[v].sort()

    # Return the neighbors of a node
    def neighbors(self, node):
        return self.adj_list[node]


# This random_walk funtion is a hint
# Refer to this function to implement the node2vec_walk function below
def random_walk(graph, start, length=5):
    # Random Walk
    walk = [start]
    for _ in range(length - 1):
        neighbors = graph.neighbors(walk[-1])
        if not neighbors:
            break
        walk.append(np.random.choice(neighbors))
    return walk


# Implement node2vec algorithm in DFS
# The length of the walk is fixed to 5
# When sampling next node, visit the node with the smallest index
# Note that it returns the trajectory of walker
# so that same node can be visited multiple times
def node2vec_walk_dfs(graph, start, length=5):
    visited = set()
    path = []

    def dfs_util(v):
        if len(path) == length:
            return

        visited.add(v)
        path.append(v)
        for neighbor in graph.neighbors(v):
            if neighbor not in visited:
                dfs_util(neighbor)
                path.append(v)  # Add the node again when returning back

    dfs_util(start)
    return path


# Train W1, W2 matrices using Skip-Gram
# The window size of fixed to 2, which means you should check each 2 nodes before and after the center node.
# - Ex. Assume we have walk sequence [1, 2, 3, 4, 5].
# - If center node is 3, we should consider [1, 2, 3, 4, 5] as the context nodes.
# - If center node is 2, we should consider [1, 2, 3, 4] as the context nodes.
# Repeat the training process for 3 epochs, with learning rate 0.01
# Use softmax function when computing the loss
def train_skipgram(walks, n_nodes, dim=128, lr=0.01, window=2, epochs=3):
    W1 = np.random.randn(n_nodes, dim)
    W2 = np.random.randn(dim, n_nodes)

    for _ in range(epochs):
        # for i in tqdm.tqdm(range(len(walks))):
        #     ctx = walks[i][max(0, i - window) : min(i + window + 1, len(walks))]
        # for x, y in combinations(ctx, 2):
        #     prob = np.exp(W1[x - 1] @ W2)  # (n_nodes,)
        #     prob /= prob.sum()
        #     prob[y - 1] -= 1
        #     W2 -= lr * np.outer(W1[x - 1], prob)
        #     W1[x - 1] -= lr * (W2 @ prob)
        for walk in tqdm.tqdm(walks):
            for i in range(len(walk)):
                center_word = walk[i]
                context_indices = (
                    walk[max(0, i - window) : i]
                    + walk[i + 1 : min(i + window + 1, len(walk))]
                )
                context_indices = np.array(context_indices)

                # Forward pass
                h = W1[center_word - 1]
                u = np.matmul(W2.T, h)
                u = np.exp(u - np.max(u))
                y_pred = u / np.sum(u)

                # Calculate error
                e = y_pred
                e[context_indices - 1] -= 1

                # Backward pass
                grad_W2 = np.outer(h, e)
                grad_W1 = np.matmul(W2, e)

                # Update weights
                W2 -= lr * grad_W2
                W1[center_word - 1] -= lr * grad_W1

    return W1, W2


# Main function
def main():
    # Don't change this code
    # This will guarantee the same output when we test your code
    np.random.seed(1116)

    # Create graph
    graph = Graph()

    # Edges list
    # Note that the edges are undirected, and node idx starts with 1
    # ex. edges = [(1, 2), (1, 3), (2, 3), (2, 4), (3, 5), (4, 5)]
    edges = []

    # Parse edges from the command line file path ======================
    with open(sys.argv[1], "rt") as f:
        for line in f:
            edge = tuple(map(int, line.split()))
            if edge not in edges:
                edges.append(edge)
    # ====================================================================

    # Update graph
    for edge in edges:
        graph.add_edge(*edge)

    # Generate random walks on DFS/BFS
    walks_dfs = [node2vec_walk_dfs(graph, node) for node in graph.adj_list]

    # Train Skip-Gram on DFS/BFS
    embeddings_dfs = train_skipgram(walks_dfs, len(graph.adj_list))

    # Print the embeddings ===========================================
    for e in embeddings_dfs[0]:
        print(f"{e[0]:.5f}")
    # ===============================================================


if __name__ == "__main__":
    main()
