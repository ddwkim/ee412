def main():
    adj = [
        [0, 1 / 3, 1 / 3, 1 / 3],
        [1 / 2, 0, 0, 1 / 2],
        [1, 0, 0, 0],
        [0, 1 / 2, 1 / 2, 0],
    ]
    M = list(map(list, zip(*adj)))
    beta = 0.8

    # case (a)
    r = [1 / 4, 1 / 4, 1 / 4, 1 / 4]
    s = [1, 0, 0, 0]
    for _ in range(100):
        r_ = r
        r = [
            sum([M[k][j] * r[j] for j in range(len(M))])
            * beta
            for k in range(len(M))
        ]
        r = [
            r[i] + (1 - beta) * s[i] / sum(s)
            for i in range(len(M))
        ]
    print(r)

    # case (b)
    r = [1 / 4, 1 / 4, 1 / 4, 1 / 4]
    s = [1, 0, 1, 0]
    for _ in range(100):
        r_ = r
        r = [
            sum([M[k][j] * r[j] for j in range(len(M))])
            * beta
            for k in range(len(M))
        ]
        r = [
            r[i] + (1 - beta) * s[i] / sum(s)
            for i in range(len(M))
        ]
    print(r)


if __name__ == "__main__":
    main()
