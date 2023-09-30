import sys

THRS = 100


def main():
    # read file
    with open(sys.argv[1], "r") as f:
        lines = f.readlines()

    lines = [line.strip().split() for line in lines]
    basckets = [list(set(line)) for line in lines if line]
    names = sorted(list(set([item for bascket in basckets for item in bascket])))
    int2name = {i: item for i, item in enumerate(names)}
    name2int = {item: i for i, item in enumerate(names)}

    # get frequent items
    counts1 = [0] * len(names)
    for bascket in basckets:
        for item in bascket:
            counts1[name2int[item]] += 1

    l1 = []
    for idx, count in enumerate(counts1):
        if count >= THRS:
            l1.append(idx)

    print(len(l1))

    # get frequent pairs, use triangular matrix method
    # use offsets to calculate index
    counts2 = [0] * (len(l1) * (len(l1) - 1) // 2)
    offsets = [sum(range(len(l1) - 1, len(l1) - 1 - i, -1)) for i in range(len(l1) - 1)]
    for bascket in basckets:
        for i, item1 in enumerate(bascket):
            for item2 in bascket[i + 1 :]:
                if item1 != item2 and name2int[item1] in l1 and name2int[item2] in l1:
                    id1, id2 = l1.index(name2int[item1]), l1.index(name2int[item2])
                    if id1 > id2:
                        id1, id2 = id2, id1
                    idx = offsets[id1] + id2 - id1 - 1
                    counts2[idx] += 1
    l2 = []
    for idx, count in enumerate(counts2):
        if count >= THRS:
            l2.append((idx, count))
    l2 = sorted(l2, key=lambda x: (-x[1], x[0]))

    print(len(l2))

    # print
    for idx, count in l2[:10]:
        for i, off in enumerate(offsets[::-1]):
            if idx >= off:
                idx -= off
                id1 = len(l1) - 2 - i
                id2 = id1 + idx + 1
                break

        conf1 = count / counts1[l1[id1]]
        conf2 = count / counts1[l1[id2]]
        print(int2name[l1[id1]], int2name[l1[id2]], count, conf1, conf2, sep="\t")


if __name__ == "__main__":
    main()
