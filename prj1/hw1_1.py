import sys
from pyspark import SparkConf, SparkContext


def main():
    conf = SparkConf()
    sc = SparkContext(conf=conf)

    # parse file
    rdd = sc.textFile(sys.argv[1])
    rdd = rdd.map(lambda x: x.split())
    rdd = rdd.filter(lambda x: len(x) == 2)
    rdd = rdd.map(lambda x: (int(x[0]), [int(y) for y in x[1].split(",")]))

    # get direct friend
    direct_friend = rdd.flatMap(
        lambda x: [((x[0], y) if x[0] < y else (y, x[0]), None) for y in x[1]]
    )

    def getPairs(x):
        pairs = []
        for i in range(len(x[1])):
            for j in range(i + 1, len(x[1])):
                pair = (x[1][i], x[1][j]) if x[1][i] < x[1][j] else (x[1][j], x[1][i])
                pairs.append((pair, [x[0]]))
        return pairs

    # get undirect friend
    undirect_friend = rdd.flatMap(getPairs)
    rdd = undirect_friend.subtractByKey(direct_friend)
    rdd = rdd.reduceByKey(lambda x, y: x + y)
    rdd = rdd.map(lambda x: (x[0], len(set(x[1]))))

    # sort and print
    results = rdd.top(10, key=lambda x: (x[1], -x[0][0], -x[0][1]))
    for result in results[:10]:
        print(result[0][0], result[0][1], result[1], sep="\t")

    sc.stop()


if __name__ == "__main__":
    main()
