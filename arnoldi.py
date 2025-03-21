import numpy as np


def arnoldi(A, u, m, tol=1e-8):
    n = len(u)
    Q = np.zeros((n, m + 1), dtype=complex)
    H = np.zeros((m + 1, m), dtype=complex)
    Q[:, 0] = u / np.linalg.norm(u)

    for j in range(m):
        Q[:, j + 1] = A @ Q[:, j]
        for i in range(j + 1):
            H[i, j] = np.vdot(Q[:, i], Q[:, j + 1])
            Q[:, j + 1] = Q[:, j + 1] - H[i, j] * Q[:, i]
        H[j + 1, j] = np.linalg.norm(Q[:, j + 1])
        if abs(H[j + 1, j]) < tol:
            return Q[:, : j + 1], H[: j + 1, : j + 1]
        Q[:, j + 1] = Q[:, j + 1] / H[j + 1, j]
    return Q, H
