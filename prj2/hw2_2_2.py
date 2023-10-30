import numpy as np


def main():
    M = [[1, 2, 3], [3, 4, 5], [5, 4, 3], [0, 2, 4], [1, 3, 5]]
    M = np.array(M)
    transpose_M = np.transpose(M)

    MtM = np.matmul(transpose_M, M)
    print("transpose(M)*M:")
    print(MtM)

    MMt = np.matmul(M, transpose_M)
    print("M*transpose(M):")
    print(MMt)

    # Find eigenpairs
    eigval_MtM, eigvec_MtM = np.linalg.eig(MtM)
    # sort eigenpairs
    idx = eigval_MtM.argsort()[::-1]
    eigval_MtM = eigval_MtM[idx]
    eigvec_MtM = eigvec_MtM[:, idx]

    eigval_MMt, eigvec_MMt = np.linalg.eig(MMt)
    # sort eigenpairs
    idx = eigval_MMt.argsort()[::-1]
    eigval_MMt = eigval_MMt[idx]
    eigvec_MMt = eigvec_MMt[:, idx]

    print("Eigenvalues of Transpose(M)*M:")
    print(np.around(eigval_MtM, decimals=3))
    print("Eigenvectors of Transpose(M)*M:")
    print(np.around(eigvec_MtM, decimals=3))

    print("Eigenvalues of M*Transpose(M):")
    print(np.around(eigval_MMt, decimals=3))
    print("Eigenvectors of M*Transpose(M):")
    print(np.around(eigvec_MMt, decimals=3))

    # find SVD using above only two eigenpairs
    V = eigvec_MtM[:, :2]
    S = np.sqrt(np.diag(eigval_MtM[:2]))

    # calculate U using V and S
    U = np.matmul(np.matmul(M, V), np.linalg.inv(S))

    print("U:")
    print(np.around(U, decimals=3))
    print("S:")
    print(np.around(S, decimals=3))
    print("V:")
    print(np.around(V, decimals=3))
    print("U*S*V^T:")
    print(np.around(np.matmul(np.matmul(U, S), V.T), decimals=3))
    print("M:")
    print(np.around(M, decimals=3))

    # rank 1 approximation
    U1 = U[:, :1]
    S1 = S[:1, :1]
    V1 = V[:, :1]
    M1 = np.matmul(np.matmul(U1, S1), V1.T)
    print("rank 1 approximation of M:")
    print(np.around(M1, decimals=3))

    print("energy of the original singular values:")
    print(np.around(np.sum(S**2), decimals=3))
    print("energy of the one-dimensional approximation:")
    print(np.around(np.sum(S1**2), decimals=3))


if __name__ == "__main__":
    main()
