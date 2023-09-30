import sys
import re
import numpy as np

REEXP = r"[A-Za-z\s]+"
THRS = 0.9
B = 6
R = 20
N = B * R


def isprime(n):
    """
    check if n is a prime number
    """
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True


def gethash(c):
    """
    get a hash function
    """
    a = np.random.randint(1, c)
    b = np.random.randint(0, c)
    return lambda x: (a * x + b) % c


def main():
    # read file
    with open(sys.argv[1], "r") as f:
        lines = f.readlines()

    articles = [line.split(maxsplit=1) for line in lines if line]
    articles = [[doc[0], re.sub(REEXP, "", doc[1]).lower()] for doc in articles]
    articles = [[doc[0], " ".join(doc[1].split())] for doc in articles]

    # get shigles
    shigles = []
    shiglesall = set()
    for doc in articles:
        shigles.append(set())
        for i in range(len(doc[1]) - 2):
            shigles[-1].add(doc[1][i : i + 3])
        shiglesall = shiglesall.union(shigles[-1])

    shiglesall = sorted(list(shiglesall))
    shigle2id = {item: i for i, item in enumerate(shiglesall)}

    c = len(shiglesall)
    while not isprime(c):
        c += 1

    # get signatures
    hashfuncs = [gethash(c) for _ in range(N)]
    signatures = []
    for doc in shigles:
        signatures.append([])
        for i in range(N):
            minhash = c + 1
            for shingle in shiglesall:
                if shingle in doc:
                    minhash = min(minhash, hashfuncs[i](shigle2id[shingle]))
            signatures[-1].append(minhash)

    # get candidates
    candidates = []
    for i in range(len(signatures)):
        for j in range(i + 1, len(signatures)):
            for b in range(B):
                if (
                    signatures[i][b * R : (b + 1) * R]
                    == signatures[j][b * R : (b + 1) * R]
                ):
                    candidates.append((i, j))
                    break

    # get results
    results = []
    for i, j in candidates:
        sigi, sigj = signatures[i], signatures[j]
        sim = sum([sigi[k] == sigj[k] for k in range(N)]) / N
        if sim >= THRS:
            results.append((articles[i][0], articles[j][0]))

    for article1, article2 in results:
        print(article1, article2, sep="\t")


if __name__ == "__main__":
    main()
