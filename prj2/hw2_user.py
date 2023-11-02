import numpy as np
import sys
import re

# Take average of top-k similar user's ratings
topk_users_to_average = 300

topk_genres = 10
svd_k = 64
alpha = 4

np.random.seed(2)


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


def parse_movie_names(file_name):
    with open(file_name) as f:
        lines = f.readlines()

    lines = [line.split(",") for line in lines]
    yearpat = re.compile(r"\((\d+)\)")
    lines = [
        [
            line[0],
            line[1],
            int(re.findall(yearpat, line[1])[0])
            if re.findall(yearpat, line[1])
            else None,
            line[2].strip(),
        ]
        for line in lines
    ]
    years = []
    moviedict = {}
    for line in lines:
        movieid = int(line[0])
        moviedict[movieid] = line[1:]
        if line[2] is not None and line[2] not in years:
            years.append(line[2])

    years.sort()
    year_median = years[len(years) // 2]
    for movieid, (name, year, genre) in moviedict.items():
        if year is None:
            moviedict[movieid][1] = year_median

    return moviedict


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
    total = np.count_nonzero(test_umatrix)

    user_ids, _ = np.nonzero(test_umatrix)
    user_ids = np.unique(user_ids)

    mdict = parse_movie_names(sys.argv[2])
    genres = set()
    for name, year, genre in mdict.values():
        genres.update(genre.split("|"))
    genres = list(genres)

    user_genre = np.zeros((len(user_ids), len(genres)))

    for i, uid in enumerate(user_ids):
        mids = np.nonzero(train_umatrix[uid])[0]
        for idx in mids:
            mid = index2mid[idx]
            _, _, genre = mdict[mid]
            for g in genre.split("|"):
                user_genre[i, list(genres).index(g)] += 1

    user_genre = normalize_matrix(user_genre)[0]
    genre_dists = cosine_dist_matrix(user_genre)
    user_dists = cosine_dist_matrix(train_umatrix_normed)

    dists_list = [user_dists, genre_dists]
    nums = [topk_users_to_average, topk_genres]
    nums = [min(num, dist.shape[1]) for dist, num in zip(dists_list, nums)]
    weights = []
    train_mats = [train_umatrix_normed, train_umatrix_normed_svd]

    for dist, num in zip(
        [user_dists, genre_dists],
        [topk_users_to_average, topk_genres],
    ):
        num = min(num, dist.shape[1])
        rmse = get_results(
            [dist], [num], [1], train_mats, user_means, user_stds, test_umatrix
        )
        weights.append(1 / rmse**alpha)
        print(rmse)

    rmse = get_results(
        dists_list,
        nums,
        weights,
        train_mats,
        user_means,
        user_stds,
        test_umatrix,
    )
    print(rmse)


def get_results(
    dists_list,
    nums,
    weights,
    train_mats,
    user_means,
    user_stds,
    test_umatrix,
):
    total = np.count_nonzero(test_umatrix)

    user_ids, _ = np.nonzero(test_umatrix)
    user_ids = np.unique(user_ids)

    rmse = 0
    import tqdm

    for uid in tqdm.tqdm(user_ids):
        mids = np.nonzero(test_umatrix[uid])[0]

        # Initialize aggregated ratings and counters for each mid
        aggregated_ratings = np.zeros(len(mids))

        for dists, weight, num in zip(dists_list, weights, nums):
            topk_users = np.argpartition(dists, num)[:, :num]
            topk_users_i = topk_users[uid]

            for train_mat in train_mats:
                # Calculate mean rating for current dist, weighted by its weight
                topk_rated_item_num = np.count_nonzero(
                    train_mat[topk_users_i][:, mids], axis=0
                )

                mean_ratings = np.divide(
                    np.sum(train_mat[topk_users_i][:, mids], axis=0),
                    topk_rated_item_num,
                    out=np.zeros_like(
                        np.sum(train_mat[topk_users_i][:, mids], axis=0),
                        dtype=float,
                    ),
                    where=topk_rated_item_num != 0,
                )

                # Accumulate weighted ratings and counters
                aggregated_ratings += mean_ratings * weight / len(train_mats)

        weighted_mean_ratings = aggregated_ratings / np.sum(weights)
        weighted_mean_ratings = weighted_mean_ratings * user_stds[uid] + user_means[uid]

        rmse += np.sum((test_umatrix[uid, mids] - weighted_mean_ratings) ** 2)
    rmse = np.sqrt(rmse / total)

    return rmse


def main():
    umatrix, uid2index, mid2index = get_matrix(sys.argv[1])
    umatrix_normed, user_means, user_stds = normalize_matrix(umatrix)

    train_umatrix = split_dataset(umatrix, 0.1)
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
