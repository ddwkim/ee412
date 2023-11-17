import sys

from pyspark import SparkConf, SparkContext

BETA = 0.9
NUM_ITER = 50
NUM_PRINT = 10


def compute_contribs(dests, rank):
    num_dests = len(dests)
    for dest in dests:
        yield (dest, rank / num_dests)


def pagerank(rdd):
    num_pages = rdd.count()
    ranks = rdd.map(lambda x: (x[0], 1 / num_pages))

    for _ in range(NUM_ITER):
        contribs = rdd.join(ranks).flatMap(
            lambda x: compute_contribs(x[1][0], x[1][1])
        )

        ranks = contribs.reduceByKey(lambda x, y: x + y)

        ranks = ranks.mapValues(
            lambda rank: (1 - BETA) / num_pages + BETA * rank
        )

    return ranks


def main():
    conf = SparkConf()
    sc = SparkContext(conf=conf)

    rdd = sc.textFile(sys.argv[1])
    rdd = rdd.map(lambda x: x.split())

    rdd = rdd.filter(lambda x: len(x) == 2)
    rdd = rdd.map(lambda x: (x[0], [x[1]]))
    rdd = rdd.reduceByKey(lambda x, y: x + y)
    rdd = rdd.map(lambda x: (x[0], sorted(list(set(x[1])))))

    rdd = rdd.map(lambda x: (x[0], x[1], 1 / len(x[1])))

    ranks = pagerank(rdd)

    top10 = ranks.top(NUM_PRINT, key=lambda x: x[1])
    for i in range(NUM_PRINT):
        print(f"{top10[i][0]}\t{top10[i][1]:.5f}")

    sc.stop()


if __name__ == "__main__":
    main()
