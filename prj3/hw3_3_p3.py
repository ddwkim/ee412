import sys

import numpy as np


class Graph:
    def __init__(self):
        self.adj_list = {}

    def add_edge(self, u, v):
        if u not in self.adj_list:
            self.adj_list[u] = []

        if v not in self.adj_list[u]:
            self.adj_list[u].append(v)
            self.adj_list[u].sort()

        if v not in self.adj_list:
            self.adj_list[v] = []

        if u not in self.adj_list[v]:
            self.adj_list[v].append(u)
            self.adj_list[v].sort()

    def neighbors(self, node):
        return self.adj_list[node]


def node2vec_walk_dfs(graph, start, length=5):
    visited = set()
    path = []

    def recurse(v):
        if len(path) == length:
            return

        visited.add(v)
        path.append(v)
        for neighbor in graph.neighbors(v):
            if neighbor not in visited:
                recurse(neighbor)
                if len(path) == length:
                    return
                path.append(v)

    recurse(start)
    return path


def train_skipgram(walks, n_nodes, dim=128, lr=0.01, window=2, epochs=3):
    W1 = np.random.randn(n_nodes, dim)
    W2 = np.random.randn(dim, n_nodes)

    for _ in range(epochs):
        for walk in walks:
            for i in range(len(walk)):
                center_word = walk[i]
                for context_word in (
                    walk[max(i - window, 0) : i] + walk[i + 1 : i + window + 1]
                ):
                    h = W1[center_word - 1]
                    u = np.matmul(W2.T, h)
                    u = np.exp(u - np.max(u))
                    y_pred = u / np.sum(u)

                    e = y_pred
                    e[context_word - 1] -= 1

                    grad_W2 = np.outer(h, e)
                    grad_W1 = np.matmul(W2, e)

                    W2 -= lr * grad_W2
                    W1[center_word - 1] -= lr * grad_W1

    return W1, W2


def main():
    # Don't change this code
    # This will guarantee the same output when we test your code
    np.random.seed(1116)

    graph = Graph()

    edges = []

    with open(sys.argv[1], "rt") as f:
        for line in f:
            edge = tuple(map(int, line.split()))
            if edge not in edges:
                edges.append(edge)

    for edge in edges:
        graph.add_edge(*edge)

    walks_dfs = [node2vec_walk_dfs(graph, node) for node in sorted(list(graph.adj_list.keys()))]

    embeddings_dfs = train_skipgram(walks_dfs, len(graph.adj_list))

    for e in [embeddings_dfs[0][4], embeddings_dfs[0][9]]:
        print(f"{e[0]:.5f}")


if __name__ == "__main__":
    main()
