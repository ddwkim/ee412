import sys

from pyspark import SparkConf, SparkContext

BETA = 0.9
NUM_ITER = 50
NUM_PRINT = 10


def main():
    conf = SparkConf()
    sc = SparkContext(conf=conf)

    rdd = sc.textFile(sys.argv[1])
    rdd = rdd.map(lambda x: x.split())

    rdd = rdd.filter(lambda x: len(x) == 2)
    rdd = rdd.map(lambda x: (x[0], [x[1]]))
    rdd = rdd.reduceByKey(lambda x, y: x + y)
    rdd = rdd.map(lambda x: (x[0], sorted(list(set(x[1])))))

    num_nodes = rdd.count()
    ranks = rdd.map(lambda x: (x[0], 1 / num_nodes))

    for _ in range(NUM_ITER):
        inflows = rdd.join(ranks).flatMap(
            lambda x: ((dest, x[1][1] / len(x[1][0])) for dest in x[1][0])
        )

        ranks = inflows.reduceByKey(lambda x, y: x + y)

        ranks = ranks.mapValues(
            lambda rank: (1 - BETA) / num_nodes + BETA * rank
        )

    top10 = ranks.top(NUM_PRINT, key=lambda x: x[1])
    for i in range(NUM_PRINT):
        print(f"{top10[i][0]}\t{top10[i][1]:.5f}")

    sc.stop()


if __name__ == "__main__":
    main()
