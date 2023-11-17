import sys
import math

from pyspark import SparkConf, SparkContext

BETA = 0.9

conf = SparkConf()
sc = SparkContext(conf=conf)


def main():
    with open(sys.argv[1], "rt") as f:
        lines = f.readlines()

    adj = []
    for line in lines:
        edge = list(map(float, line.split()))
        if edge not in adj:
            adj.append(edge)

    rdd = sc.parallelize(adj)
    rdd = rdd.map(lambda x: (x[0], x[1]))
    rdd = rdd.groupByKey()
    rdd = rdd.mapValues(list)


if __name__ == "__main__":
    main()
