import numpy as np
import sys

# Take average of top-k similar user's ratings
topk_users_to_average = 100

svd_k = 64

np.random.seed(6)


def cosine_dist_matrix(user_features):
    # Calculate cosine distance for binary matrices
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


def split_dataset(umatrix, ratio):
    """
    Split the dataset into training and testing sets

    """

    r, c = np.nonzero(umatrix)
    num_nonzero = len(r)
    num_test = int(num_nonzero * ratio)
    num_test = max(num_test, umatrix.shape[0])

    print(f"number of test samples are {num_test}", file=sys.stderr)
    # make sure to include all users
    train_umatrix = umatrix.copy()
    for i in range(umatrix.shape[0]):
        nonzero = np.nonzero(umatrix[i])[0]
        train_umatrix[i, nonzero[0]] = 0
    num_test -= umatrix.shape[0]

    rc = [[i, j] for i, j in zip(r, c)]

    np.random.shuffle(rc)
    rc = rc[:num_test]
    r = [i for i, j in rc]
    c = [j for i, j in rc]

    train_umatrix[r, c] = 0

    return train_umatrix


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

    return umatrix, uid2index, mid2index


def normalize_matrix(umatrix):
    nnz_cols = np.count_nonzero(umatrix, axis=1)
    row_sums = np.sum(umatrix, axis=1)
    row_means = np.divide(
        row_sums, nnz_cols, out=np.zeros_like(row_sums), where=nnz_cols != 0
    )

    # subtract mean from nonzero columns
    umatrix_normed = umatrix.copy()
    user_stds = np.ones((umatrix.shape[0],))
    for i, row in enumerate(umatrix_normed):
        nonzero_indices = np.nonzero(row)[0]
        umatrix_normed[i, nonzero_indices] -= row_means[i]

    return umatrix_normed, row_means, user_stds


def perform_svd(umatrix, k):
    u, s, vh = np.linalg.svd(umatrix, full_matrices=False)
    s = np.diag(s)

    # s = s + np.eye(s.shape[0]) * s[k, k]

    umatrix_svd = np.dot(np.dot(u[:, :k], s[:k, :k]), vh[:k, :])
    return umatrix_svd


def user_based(
    umatrix,
    train_umatrix,
    uid2index,
    mid2index,
    user_means,
    user_stds,
):
    """
    INPUT: utility matrix, user id
    OUTPUT: top k recommended items

    DESCRIPTION:
    Returns the top recommendations using user-based collaborative
    filtering.
    """
    index2mid = {i: mid for mid, i in mid2index.items()}

    train_umatrix_normed, user_means, user_stds = normalize_matrix(train_umatrix)
    train_umatrix_normed_svd = perform_svd(train_umatrix_normed, svd_k)

    test_umatrix = umatrix - train_umatrix

    user_ids, _ = np.nonzero(test_umatrix)
    user_ids = np.unique(user_ids)

    user_dists = cosine_dist_matrix(train_umatrix_normed)

    train_mats = [train_umatrix_normed, train_umatrix_normed_svd]

    rmse = get_results(
        user_dists,
        topk_users_to_average,
        train_mats,
        user_means,
        test_umatrix,
    )
    print(rmse)


def get_results(
    dists,
    num,
    train_mats,
    user_means,
    test_umatrix,
):
    total = np.count_nonzero(test_umatrix)

    user_ids, movie_ids = np.nonzero(test_umatrix)
    user_ids = np.unique(user_ids)
    topk_users = np.argpartition(dists, num)[:, :num]

    train_mats = np.stack(train_mats, axis=-1)

    rmse = 0
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

        rmse += np.sum((test_umatrix[uid, mids] - mean_ratings) ** 2)
    rmse = np.sqrt(rmse / total)

    return rmse


def main():
    umatrix, uid2index, mid2index = get_matrix(sys.argv[1])
    umatrix_normed, user_means, user_stds = normalize_matrix(umatrix)

    train_umatrix = split_dataset(umatrix, 0.05)
    train_umatrix_normed, user_means, user_stds = normalize_matrix(train_umatrix)

    user_based(
        umatrix,
        train_umatrix,
        uid2index,
        mid2index,
        user_means,
        user_stds,
    )


if __name__ == "__main__":
    main()
