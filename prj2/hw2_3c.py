import sys
import os

import numpy as np

# Take average of top-k similar user's ratings
topk_users_to_average = 100

svd_k = 64


def cosine_dist_matrix(user_features):
    """
    Calculate cosine distance
    """

    norms = np.linalg.norm(user_features, axis=1)
    norms = norms[:, None]
    cosine_similarity = np.dot(user_features, user_features.T)
    cosine_similarity = np.divide(
        cosine_similarity,
        norms * norms.T,
        out=np.zeros_like(cosine_similarity),
        where=norms * norms.T != 0,
    )
    np.fill_diagonal(cosine_similarity, -1)

    return 1 - cosine_similarity


def get_matrix(file_name):
    """
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

    return umatrix, uid2index, mid2index


def parse_test_file(file_name, uid2index, mid2index):
    with open(file_name) as f:
        lines = f.readlines()

    test_matrix = np.zeros((len(uid2index), len(mid2index)))
    for line in lines:
        uid, mid, *_ = line.split(",")
        uid = int(uid)
        mid = int(mid)

        # movie not in training set is ignored
        if int(mid) not in mid2index:
            continue

        test_matrix[uid2index[uid], mid2index[mid]] = 1

    return test_matrix


def write_prediction(
    test_file_name, result_file_name, test_matrix, uid2index, mid2index
):
    with open(test_file_name) as f:
        lines = f.readlines()

    with open(result_file_name, "w") as f:
        for line in lines:
            uid, mid, _, time = line.split(",")
            # movie not in training set is ignored
            if int(mid) not in mid2index:
                continue
            rating = test_matrix[uid2index[int(uid)], mid2index[int(mid)]]
            f.write(f"{uid},{mid},{rating},{time}")

    return


def normalize_matrix(umatrix):
    nnz_cols = np.count_nonzero(umatrix, axis=1)
    row_sums = np.sum(umatrix, axis=1)
    row_means = np.divide(
        row_sums, nnz_cols, out=np.zeros_like(row_sums), where=nnz_cols != 0
    )

    # subtract mean from nonzero columns
    umatrix_normed = umatrix.copy()
    for i, row in enumerate(umatrix_normed):
        nonzero_indices = np.nonzero(row)[0]
        umatrix_normed[i, nonzero_indices] -= row_means[i]

    return umatrix_normed, row_means


def perform_svd(umatrix, k):
    u, s, vh = np.linalg.svd(umatrix, full_matrices=False)
    s = np.diag(s)

    umatrix_svd = np.dot(np.dot(u[:, :k], s[:k, :k]), vh[:k, :])
    return umatrix_svd


def predict(
    dists,
    num,
    train_mats,
    user_means,
    test_umatrix,
):
    user_ids, _ = np.nonzero(test_umatrix)
    user_ids = np.unique(user_ids)
    topk_users = np.argpartition(dists, num)[:, :num]

    train_mats = np.stack(train_mats, axis=-1)

    for uid in user_ids:
        mids = np.nonzero(test_umatrix[uid])[0]

        topk_users_i = topk_users[uid]

        topk_rated_item_num = np.count_nonzero(
            train_mats[topk_users_i][:, mids], axis=0
        )

        mean_ratings = np.divide(
            np.sum(train_mats[topk_users_i][:, mids], axis=0),
            topk_rated_item_num,
            out=np.zeros_like(
                np.sum(train_mats[topk_users_i][:, mids], axis=0),
                dtype=float,
            ),
            where=topk_rated_item_num != 0,
        )

        mean_ratings = (
            mean_ratings.sum(axis=-1) / train_mats.shape[-1] + user_means[uid]
        )

        test_umatrix[uid, mids] = mean_ratings

    return test_umatrix


def main():
    umatrix, uid2index, mid2index = get_matrix(sys.argv[1])
    umatrix_normed, user_means = normalize_matrix(umatrix)
    umatrix_normed_svd = perform_svd(umatrix_normed, svd_k)

    user_dists = cosine_dist_matrix(umatrix_normed)
    train_mats = [umatrix_normed, umatrix_normed_svd]

    test_umatrix = parse_test_file(sys.argv[2], uid2index, mid2index)

    test_umatrix = predict(
        user_dists,
        topk_users_to_average,
        train_mats,
        user_means,
        test_umatrix,
    )

    write_prediction(
        sys.argv[2],
        os.path.join(os.getcwd(), "output.txt"),
        test_umatrix,
        uid2index,
        mid2index,
    )


if __name__ == "__main__":
    main()
