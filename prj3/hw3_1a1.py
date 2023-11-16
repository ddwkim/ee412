def main():
    adj = [
        [1 / 3, 1 / 3, 1 / 3],
        [1 / 2, 0, 1 / 2],
        [0, 1 / 2, 1 / 2],
    ]
    M = list(map(list, zip(*adj)))
    beta = 0.8
    M = [
        [beta * x + (1 - beta) / len(adj) for x in row]
        for row in M
    ]

    print(M)
    r = [1 / 3, 1 / 3, 1 / 3]
    for _ in range(100):
        r = [
            sum([M[k][j] * r[j] for j in range(len(M))])
            for k in range(len(M))
        ]
    print(r)
    Mr = [
        sum([M[i][j] * r[j] for j in range(len(M))])
        for i in range(len(M))
    ]
    print(Mr)


if __name__ == "__main__":
    main()
