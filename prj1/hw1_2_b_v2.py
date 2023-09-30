import sys

THRS = 100


def main():
    with open(sys.argv[1], "r") as f:
        lines = f.readlines()

    lines = [line.strip().split() for line in lines]
    basckets = [list(set(line)) for line in lines if line]
    names = sorted(list(set([item for bascket in basckets for item in bascket])))
    int2name = {i: item for i, item in enumerate(names)}
    name2int = {item: i for i, item in enumerate(names)}

    single2count = {}
    for bascket in basckets:
        for item in bascket:
            if name2int[item] not in single2count:
                single2count[name2int[item]] = 0
            single2count[name2int[item]] += 1

    print(sum([count >= THRS for count in single2count.values()]))

    pair2count = {}
    for bascket in basckets:
        for i, item1 in enumerate(bascket):
            for item2 in bascket[i + 1 :]:
                if item1 != item2:
                    id1, id2 = name2int[item1], name2int[item2]
                    if single2count[id1] >= THRS and single2count[id2] >= THRS:
                        pair = (id1, id2) if id1 < id2 else (id2, id1)
                        if pair not in pair2count:
                            pair2count[pair] = 0
                        pair2count[pair] += 1

    print(sum([count >= THRS for count in pair2count.values()]))

    results = sorted(
        [(pair, count) for pair, count in pair2count.items()],
        key=lambda x: (-x[1], x[0][0], x[0][1]),
    )
    for (id1, id2), count in results[:10]:
        conf1 = count / single2count[id1]
        conf2 = count / single2count[id2]
        print(int2name[id1], int2name[id2], count, conf1, conf2, sep="\t")


if __name__ == "__main__":
    main()
