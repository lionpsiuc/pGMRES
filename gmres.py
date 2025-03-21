from scipy.linalg import lstsq

import matplotlib.pyplot as plt
import numpy as np


def gmres(A, b, m):
    """Implements the GMRES algorithm.

    This function solves a linear system using the GMRES algorithm. The
    algorithm constructs an orthonormal basis for the Krylov subspace and
    minimises the residual norm in this subspace.

    Args:
        A (numpy.ndarray): Square matrix of the linear system.
        b (numpy.ndarray): Right-hand side vector of the linear system.
        m (int): Maximum number of iterations to perform.

    Returns:
        tuple:
            - x (numpy.ndarray): Approximate solution to Ax = b.
            - residuals (numpy.ndarray): Array containing the residual norms at
                                         each iteration.
    """

    # Define the product function
    matrix_A = lambda x: A @ x

    # Get dimension of the system
    n = len(b)

    # Initialize matrices for Arnoldi iteration
    Q = np.zeros((n, m + 1))  # Orthonormal basis vectors
    H = np.zeros((m + 1, m))  # Upper Hessenberg matrix

    # Start with zero initial guess
    x0 = np.zeros(n)

    # Compute initial residual
    r0 = b - matrix_A(x0)
    beta = np.linalg.norm(r0)

    # First basis vector is the normalised residual
    Q[:, 0] = r0 / beta

    # Initialise residuals history with initial residual norm
    residuals = [beta]

    for j in range(m):
        q = matrix_A(Q[:, j])

        # Orthogonalise against previous basis vectors
        for i in range(j + 1):
            H[i, j] = np.dot(Q[:, i], q)  # Compute projection
            q = q - H[i, j] * Q[:, i]  # Orthogonalise

        H[j + 1, j] = np.linalg.norm(q)

        # Avoid division by zero for numerical stability
        if H[j + 1, j] > 1e-12:
            Q[:, j + 1] = q / H[j + 1, j]  # Normalise the new basis vector

        # Set up the right-hand side for the least squares problem
        e1 = np.zeros(j + 2)
        e1[0] = beta

        # Solve the least squares problem to minimise the residual
        y, residual, rank, s = lstsq(H[: j + 2, : j + 1], e1)

        res = np.linalg.norm(beta * e1[: j + 2] - H[: j + 2, : j + 1] @ y)
        residuals.append(res)

    # Compute the final solution using the least squares solution
    x = x0 + Q[:, :m] @ y

    return x, np.array(residuals)


def create_matrix_A(n):
    """Creates a tridiagonal matrix (of size n x n) with a strcture as per the
    assignment instructions.

    Args:
        n (int): Size of the square matrix.

    Returns:
        numpy.ndarray: The tridiagonal matrix (of size n x n).
    """
    A = np.zeros((n, n))
    for i in range(n):
        A[i, i] = -4  # Diagonal
        if i > 0:
            A[i, i - 1] = 1  # Lower diagonal
        if i < n - 1:
            A[i, i + 1] = 1  # Upper diagonal
    return A


def create_vector_b(n):
    """Creates the right-hand side vector as per the assignment instructions.

    Args:
        n (int): Length of the vector.

    Returns:
        numpy.ndarray: The right-hand side vector (of length n).
    """
    b = np.zeros(n)
    for i in range(n - 1):
        b[i] = (i + 1) / n  # Set elements as per the assignment instructions
    b[n - 1] = 1
    return b


def run_gmres_experiment():
    """Runs the GMRES algorithm for different matrix sizes and plots the
    convergence.
    """

    # Different matrix sizes to test
    n_values = [8, 16, 32, 64, 128, 256]
    plt.figure(figsize=(10, 6))

    # Run GMRES for each matrix size
    for n in n_values:
        A = create_matrix_A(n)
        b = create_vector_b(n)
        m = n // 2
        x, residuals = gmres(A, b, m)
        b_norm = np.linalg.norm(b)
        relative_residuals = residuals / b_norm

        # Plot the convergence curve
        plt.semilogy(relative_residuals, label=f"n = {n}")

    plt.title("Convergence of the GMRES Algorithm for Different Matrix Sizes")
    plt.xlabel(r"Iteration $m=\frac{n}{2}$")
    plt.ylabel(r"Relative Residual $\frac{\|r_k\|_2}{\|b\|_2}$")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("convergence.png")


run_gmres_experiment()
