import numpy as np


def arnoldi(A, u, m):
    """Implements the Arnoldi iteration algorithm for Krylov subspaces.

    This function performs the Arnoldi iteration to build an orthonormal basis
    for the Krylov subspace of degree m for the matrix A with starting vector u.
    It uses the modified Gram-Schmidt process for orthogonalisation.

    Args:
        A (numpy.ndarray): A square matrix (of size n x n).
        u (numpy.ndarray): The starting vector (of length n).
        m (int): The dimension of the Krylov subspace to construct.

    Returns:
        tuple:
            - Q (numpy.ndarray): A matrix (of size n x (m + 1)) whose columns
                                 form an orthonormal basis for the Krylov
                                 subspace.
            - H (numpy.ndarray): An upper Hessenberg matrix (of size (m + 1) x
                                 m) that represents the projection of A onto
                                 the Krylov subspace.
    """

    # Length of our starting vector
    n = len(u)

    # Initialise the orthonormal basis and the upper Hessenberg matrix
    Q = np.zeros((n, m + 1), dtype=complex)
    H = np.zeros((m + 1, m), dtype=complex)

    # Normalise the intial vector to create the first basis vector
    Q[:, 0] = u / np.linalg.norm(u)

    # Constructing the Krlov subspace
    for j in range(m):

        # Construct the next basis vector (i.e., A * q_j)
        Q[:, j + 1] = A @ Q[:, j]

        # Orthogonalise the new basis vector against the previous ones
        for i in range(j + 1):
            H[i, j] = np.vdot(Q[:, i], Q[:, j + 1])
            Q[:, j + 1] = Q[:, j + 1] - H[i, j] * Q[:, i]

        # Normalise the new basis vector
        H[j + 1, j] = np.linalg.norm(Q[:, j + 1])
        Q[:, j + 1] = Q[:, j + 1] / H[j + 1, j]

    return Q, H


# Both as given in the assignment
A = np.array(
    [
        [3, 8, 7, 3, 3, 7, 2, 3, 4, 8],
        [5, 4, 1, 6, 9, 8, 3, 7, 1, 9],
        [3, 6, 9, 4, 8, 6, 5, 6, 6, 6],
        [5, 3, 4, 7, 4, 9, 2, 3, 5, 1],
        [4, 4, 2, 1, 7, 4, 2, 2, 4, 5],
        [4, 2, 8, 6, 6, 5, 2, 1, 1, 2],
        [2, 8, 9, 5, 2, 9, 4, 7, 3, 3],
        [9, 3, 2, 2, 7, 3, 4, 8, 7, 7],
        [9, 1, 9, 3, 3, 1, 2, 7, 7, 1],
        [9, 3, 2, 2, 6, 4, 4, 7, 3, 5],
    ]
)
b = np.array(
    [
        0.757516242460009,
        2.734057963614329,
        -0.555605907443403,
        1.144284746786790,
        0.645280108318073,
        -0.085488474462339,
        -0.623679022063185,
        -0.465240896342741,
        2.382909057772335,
        -0.120465395885881,
    ]
)

# Perform the Arnoldi iteration
Q, H = arnoldi(A, b, 9)

# Extract the ninth column
Q_9 = Q[:, 9]

# Print the results
print(Q_9)
print("\nShape of Q:", Q.shape)
print("Shape of H:", H.shape)
