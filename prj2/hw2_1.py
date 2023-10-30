import sys
import math
import numpy as np

from pyspark import SparkConf, SparkContext

conf = SparkConf()
sc = SparkContext(conf=conf)


def dist(x, y):
    """
    INPUT: two points x and y
    OUTPUT: the Euclidean distance between two points x and y

    DESCRIPTION: Returns the Euclidean distance between two points.
    """
    x = np.array(x)
    y = np.array(y)

    return np.linalg.norm(x - y)


def parse_line(line):
    """
    INPUT: one line from input file
    OUTPUT: parsed line with numerical values

    DESCRIPTION: Parses a line to coordinates.
    """
    line = line.split()
    line = [float(x) for x in line]

    return line


def pick_points(k):
    """
    INPUT: value of k for k-means algorithm
    OUTPUT: the list of initial k centroids.

    DESCRIPTION: Picks the initial cluster centroids for running k-means.
    """
    centroids = []
    with open(sys.argv[1]) as f:
        lines = f.readlines()
    lines = [parse_line(line) for line in lines]
    centroids.append(lines[0])

    for _ in range(1, k):
        max_dist = 0
        max_index = 0
        for i in range(len(lines)):
            min_dist = math.inf
            for centroid in centroids:
                distance = dist(lines[i], centroid)
                min_dist = min(distance, min_dist)
            if min_dist > max_dist:
                max_dist = min_dist
                max_index = i
        centroids.append(lines[max_index])

    return centroids


def assign_cluster(centroids, point):
    """
    INPUT: list of centorids and a point
    OUTPUT: a pair of (closest centroid, given point)

    DESCRIPTION: Assigns a point to the closest centroid.
    """
    dists = [dist(point, centroid) for centroid in centroids]
    min_dist = min(dists)
    min_index = dists.index(min_dist)

    return min_index, point


def compute_diameter(cluster):
    """
    INPUT: cluster
    OUTPUT: diameter of the given cluster

    DESCRIPTION: Computes the diameter of a cluster.
    """
    diameter = 0
    for i in range(len(cluster)):
        for j in range(i + 1, len(cluster)):
            distance = dist(cluster[i], cluster[j])
            if distance > diameter:
                diameter = distance

    return diameter


def kmeans(centroids):
    """
    INPUT: list of centroids
    OUTPUT: average diameter of the clusters

    DESCRIPTION:
    Runs the k-means algorithm and computes the cluster diameters.
    Returns the average diameter of the clusters.

    You may use PySpark things at this function.
    """
    with open(sys.argv[1]) as f:
        lines = f.readlines()
    lines = [parse_line(line) for line in lines]

    rdd = sc.parallelize(lines)
    rdd = rdd.map(lambda x: assign_cluster(centroids, x))
    rdd = rdd.groupByKey()
    rdd = rdd.map(lambda x: list(x[1]))
    rdd = rdd.map(lambda x: compute_diameter(x))
    rdd = rdd.collect()
    average_diameter = sum(rdd) / len(rdd)

    return average_diameter


if __name__ == "__main__":
    k = int(sys.argv[2])
    centroids = pick_points(k)
    average_diameter = kmeans(centroids)
    print(average_diameter)
