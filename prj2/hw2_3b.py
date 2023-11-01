import sys
import numpy as np

# Take average of top-k similar user's ratings
topk_users_to_average = 10
# Take average of top-k similar items ratings
topk_items_to_average = 10
# Considering items 1 to 1000
num_items_for_prediction = 1000
# Top-k predictions of items with highest ratings
topk_items = 5
# Target user's id
target_user_id = 600


def get_matrix(file_name):
    """
    INPUT: file name
    OUTPUT: utility matrix from the file

    DESCRIPTION:
    Reads the utility matrix from the file.
    """

    with open(file_name) as f:
        lines = f.readlines()

    lines = [list(map(float, line.split(",")[:3])) for line in lines]
    lines = np.array(lines)

    uids = len(np.unique(lines[:, 0]))
    mids = len(np.unique(lines[:, 1]))
    umatrix = np.zeros((uids, mids))
    uid2index = {uid: i for i, uid in enumerate(np.unique(lines[:, 0]))}
    mid2index = {mid: i for i, mid in enumerate(np.unique(lines[:, 1]))}

    indices = lines[:, :2].astype(int)
    ratings = lines[:, 2].astype(float)

    indices = list(map(lambda x: [uid2index[x[0]], mid2index[x[1]]], indices))
    indices = np.array(indices)

    umatrix[indices[:, 0], indices[:, 1]] = ratings
    # for each row, count nonzero columns
    nnz_cols = np.count_nonzero(umatrix, axis=1)
    # for each row, sum nonzero columns
    row_sums = np.sum(umatrix, axis=1)
    # for each row, subtract mean from nonzero columns
    row_means = np.divide(
        row_sums, nnz_cols, out=np.zeros_like(row_sums), where=nnz_cols != 0
    )

    # subtract mean from nonzero columns
    umatrix_normed = umatrix.copy()
    for i, row in enumerate(umatrix_normed):
        nonzero_indices = np.nonzero(row)[0]
        umatrix_normed[i, nonzero_indices] -= row_means[i]

    return umatrix, umatrix_normed, uid2index, mid2index


def user_based(umatrix, umatrix_normed, uid2index, mid2index, user_id):
    """
    INPUT: utility matrix, user id
    OUTPUT: top k recommended items

    DESCRIPTION:
    Returns the top recommendations using user-based collaborative
    filtering.
    """
    index2mid = {i: mid for mid, i in mid2index.items()}
    # get index of mid whose mid is 1000 or under
    for idx, mid in sorted(index2mid.items(), key=lambda x: x[1]):
        if mid > num_items_for_prediction:
            break
    max_idx = idx

    uid = uid2index[user_id]
    sims = np.dot(umatrix_normed, umatrix_normed[uid])
    norms = np.linalg.norm(umatrix_normed, axis=1) * np.linalg.norm(umatrix_normed[uid])

    sims = np.divide(
        sims,
        norms,
        out=-np.ones_like(sims, dtype=float),
        where=norms != 0,
    )
    sims[uid] = -1
    dists = 1 - sims
    topk_users = np.argsort(dists)[:topk_users_to_average]

    # Pre-compute the ratings for the user of interest
    user_ratings = umatrix[uid, :max_idx]

    # Find the indices where the user has already rated
    already_rated = np.where(user_ratings != 0)[0]
    not_rated = np.setdiff1d(np.arange(max_idx), already_rated)
    topk_rated_item_num = np.count_nonzero(umatrix[topk_users][:, not_rated], axis=0)

    mean_ratings = np.divide(
        np.sum(umatrix[topk_users][:, not_rated], axis=0),
        topk_rated_item_num,
        out=np.zeros_like(
            np.sum(umatrix[topk_users][:, not_rated], axis=0), dtype=float
        ),
        where=topk_rated_item_num != 0,
    )

    # Combine the already rated and computed mean ratings
    all_ratings = np.zeros(max_idx)
    all_ratings[already_rated] = user_ratings[already_rated]
    all_ratings[not_rated] = mean_ratings

    all_ratings = sorted(
        [(rating, index2mid[midx]) for midx, rating in enumerate(all_ratings)],
        reverse=True,
        key=lambda x: (x[0], -x[1]),
    )

    for rating, mid in all_ratings[:topk_items]:
        print(f"{int(mid)}\t{rating}")

    return


def item_based(umatrix, umatrix_normed, uid2index, mid2index, user_id):
    """
    INPUT: utility matrix, user_id
    OUTPUT: top k recommended items

    DESCRIPTION:
    Returns the top recommendations using item-based collaborative
    filtering.
    """
    index2mid = {i: mid for mid, i in mid2index.items()}
    # get index of mid whose mid is 1000 or under
    for idx, mid in sorted(index2mid.items(), key=lambda x: x[1]):
        if mid > num_items_for_prediction:
            break
    max_idx = idx

    uid = uid2index[user_id]
    # dimension for sims is (max_idx, rest of the items)
    sims = np.matmul(umatrix_normed[:, :max_idx].T, umatrix_normed[:, max_idx:])
    norms = np.outer(
        np.linalg.norm(umatrix_normed[:, :max_idx], axis=0),
        np.linalg.norm(umatrix_normed[:, max_idx:], axis=0),
    )
    sims = np.divide(
        sims,
        norms,
        out=-np.ones_like(sims, dtype=float),
        where=norms != 0,
    )
    dists = 1 - sims
    topk_indices = np.argsort(dists, axis=0)[:topk_items_to_average]

    umatrix_topk = umatrix[uid, topk_indices + max_idx]
    topk_rated_item_num = np.count_nonzero(umatrix_topk, axis=0)
    mean_rating = np.divide(
        np.sum(umatrix_topk, axis=0),
        topk_rated_item_num,
        out=np.zeros_like(topk_rated_item_num, dtype=float),
        where=topk_rated_item_num != 0,
    )

    predicted_ratings = sorted(
        [(rating, index2mid[midx]) for midx, rating in enumerate(mean_rating)],
        reverse=True,
        key=lambda x: (x[0], -x[1]),
    )

    for rating, mid in predicted_ratings[:topk_items]:
        print(f"{int(mid)}\t{rating}")

    return


def main():
    umatrix, umatrix_normed, uid2index, mid2index = get_matrix(sys.argv[1])
    user_based(umatrix, umatrix_normed, uid2index, mid2index, target_user_id)
    item_based(umatrix, umatrix_normed, uid2index, mid2index, target_user_id)


if __name__ == "__main__":
    main()
