import numpy as np

from hw3_3_p3 import Graph, node2vec_walk_dfs, train_skipgram


# Main test function. Don't change this function
def main():
    # Don't change this code
    # This will guarantee the same output when we test your code
    np.random.seed(1116)

    # Create graph
    graph = Graph()

    # Edges list
    # This is a small test graph :)
    edges = [(1, 2), (1, 3), (2, 4), (2, 5), (3, 6), (3, 7)]

    # Update graph
    for edge in edges:
        graph.add_edge(*edge)

    # Generate random walks on DFS
    walks_dfs = [node2vec_walk_dfs(graph, node) for node in graph.adj_list]

    print("=====================================")
    print("Test of node2vec_walk_dfs()")
    for i, item in enumerate(walks_dfs):
        print("node : ", i + 1)
        print("walk : ", item)
    print("=====================================\n")

    # Train Skip-Gram on DFS
    W1, W2 = train_skipgram(walks_dfs, len(graph.adj_list))

    print("=====================================")
    print("Test of train_skipgram()")
    print(
        "First element of node 5's embedding: ",
    )
    print(f"{W1[4][0]:.5f}")
    print("=====================================\n")


if __name__ == "__main__":
    main()
