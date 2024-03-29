import numpy as np


def power_iteration(A, B, nsim):
    # Choose a random starting vector
    b_k = B

    for _ in range(nsim):
        # Calculate the matrix-by-vector product Ab
        b_k1 = np.dot(A, b_k)

        # Re normalize the vector
        b_k = b_k1 / np.linalg.norm(b_k1)

    return b_k


def main():
    A = np.array([[1, 1, 1], [1, 2, 3], [1, 3, 6]])
    B = np.array([1, 1, 1])

    eigvec = power_iteration(A, B, nsim=100)
    eigval = np.dot(np.dot(A, eigvec), eigvec)

    print("eigvec: ", np.around(eigvec, decimals=3))
    print("eigval: ", np.around(eigval, decimals=3))

    # second eigenvalue
    A1 = A - eigval * np.outer(eigvec, eigvec)
    print("A: ", np.around(A1, decimals=3))

    eigvec = power_iteration(A1, B, nsim=100)
    eigval = np.dot(np.dot(A1, eigvec), eigvec)

    print("eigvec: ", np.around(eigvec, decimals=3))
    print("eigval: ", np.around(eigval, decimals=3))

    # third eigenvalue
    A2 = A1 - eigval * np.outer(eigvec, eigvec)
    print("A: ", np.around(A2, decimals=3))

    eigvec = power_iteration(A2, B, nsim=100)
    eigval = np.dot(np.dot(A2, eigvec), eigvec)

    print("eigvec: ", np.around(eigvec, decimals=3))
    print("eigval: ", np.around(eigval, decimals=3))


if __name__ == "__main__":
    main()
