import sys

THRS = 100


def main():
    # read file
    with open(sys.argv[1], "r") as f:
        lines = f.readlines()

    lines = [line.strip().split() for line in lines]
    basckets = [line for line in lines if line]
    names = sorted(list(set([item for bascket in basckets for item in bascket])))
    int2name = {i: item for i, item in enumerate(names)}
    name2int = {item: i for i, item in enumerate(names)}

    # get frequent items
    single2count = {}
    for bascket in basckets:
        for item in bascket:
            if name2int[item] not in single2count:
                single2count[name2int[item]] = 0
            single2count[name2int[item]] += 1

    # get frequent pairs
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

    # get frequent triples
    triple2count = {}
    for bascket in basckets:
        for i, item1 in enumerate(bascket):
            for j, item2 in enumerate(bascket[i + 1 :]):
                for item3 in bascket[i + j + 2 :]:
                    if item1 != item2 and item1 != item3 and item2 != item3:
                        id1, id2, id3 = (
                            name2int[item1],
                            name2int[item2],
                            name2int[item3],
                        )
                        id1, id2, id3 = sorted([id1, id2, id3])
                        if (
                            (id1, id2) in pair2count
                            and pair2count[(id1, id2)] >= THRS
                            and (id1, id3) in pair2count
                            and pair2count[(id1, id3)] >= THRS
                            and (id2, id3) in pair2count
                            and pair2count[(id2, id3)] >= THRS
                        ):
                            triple = tuple(sorted([id1, id2, id3]))
                            if triple not in triple2count:
                                triple2count[triple] = 0
                            triple2count[triple] += 1

    # print
    results = sorted(
        [(triple, count) for triple, count in triple2count.items()],
        key=lambda x: (-x[1], x[0]),
    )
    results = list(filter(lambda x: x[1] >= THRS, results))

    print(len(results))
    for (id1, id2, id3), count in results[:10]:
        conf1 = count / pair2count[(id1, id2)]
        conf2 = count / pair2count[(id1, id3)]
        conf3 = count / pair2count[(id2, id3)]
        print(
            int2name[id1],
            int2name[id2],
            int2name[id3],
            count,
            conf1,
            conf2,
            conf3,
            sep="\t",
        )


if __name__ == "__main__":
    main()
