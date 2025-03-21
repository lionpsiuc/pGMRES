from scipy.linalg import lstsq

import matplotlib.pyplot as plt
import numpy as np


def gmres(A, b, m):
    matrix_A = lambda x: A @ x
    n = len(b)
    Q = np.zeros((n, m + 1))
    H = np.zeros((m + 1, m))
    x0 = np.zeros(n)
    r0 = b - matrix_A(x0)
    beta = np.linalg.norm(r0)
    Q[:, 0] = r0 / beta
    residuals = [beta]
    for j in range(m):
        q = matrix_A(Q[:, j])
        for i in range(j + 1):
            H[i, j] = np.dot(Q[:, i], q)
            q = q - H[i, j] * Q[:, i]
        H[j + 1, j] = np.linalg.norm(q)
        if H[j + 1, j] > 1e-12:
            Q[:, j + 1] = q / H[j + 1, j]
        e1 = np.zeros(j + 2)
        e1[0] = beta
        y, residual, rank, s = lstsq(H[: j + 2, : j + 1], e1)
        res = np.linalg.norm(beta * e1[: j + 2] - H[: j + 2, : j + 1] @ y)
        residuals.append(res)
    x = x0 + Q[:, :m] @ y
    return x, np.array(residuals)


def create_matrix_A(n):
    A = np.zeros((n, n))
    for i in range(n):
        A[i, i] = -4
        if i > 0:
            A[i, i - 1] = 1
        if i < n - 1:
            A[i, i + 1] = 1
    return A


def create_vector_b(n):
    b = np.zeros(n)
    for i in range(n - 1):
        b[i] = (i + 1) / n
    b[n - 1] = 1
    return b


def run_gmres_experiment():
    n_values = [8, 16, 32, 64, 128, 256]
    plt.figure(figsize=(10, 6))
    for n in n_values:
        A = create_matrix_A(n)
        b = create_vector_b(n)
        m = n // 2
        x, residuals = gmres(A, b, m)
        b_norm = np.linalg.norm(b)
        relative_residuals = residuals / b_norm
        plt.semilogy(relative_residuals, label=f"n = {n}")
    plt.title("Convergence of the GMRES Algorithm for Different Matrix Sizes")
    plt.xlabel(r"Iteration $m=\frac{n}{2}$")
    plt.ylabel(r"Relative Residual $\frac{\|r_k\|_2}{\|b\|_2}$")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("convergence.png")


run_gmres_experiment()
