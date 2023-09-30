import re
import sys
from pyspark import SparkConf, SparkContext


def main():

    conf = SparkConf()
    sc = SparkContext(conf=conf)
    
    rdd = sc.textFile(sys.argv[1])
    rdd = rdd.flatMap(lambda l: re.split(r'[^\w]+', l))
    rdd = rdd.filter(lambda w: w and w[0].isalpha())
    rdd = rdd.map(lambda w: w.lower()).distinct()
    rdd = rdd.map(lambda w: (w[0], 1))

    base_rdd = sc.parallelize([(chr(i), 0) for i in range(ord('a'), ord('z') + 1)])
    rdd = rdd.union(base_rdd)

    rdd = rdd.reduceByKey(lambda n1, n2: n1 + n2)
    rdd = rdd.map(lambda t: t[0] + '\t' + str(t[1]))
    
    results = sorted(rdd.collect())
    for result in results:
        print(result)

    sc.stop()


if __name__ == "__main__":
    main()