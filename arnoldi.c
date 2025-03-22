/**
 * @file arnoldi.c
 *
 * @brief Implementation of the Arnoldi iteration algorithm for Krylov
 *        subspaces.
 *
 * @author Ion Lipsiuc
 * @date 2025-03-22
 * @version 1.0
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

/**
 * @brief Performs the Arnoldi iteration algorithm to build a Krylov subspace.
 *
 * Adapted from Numerical Linear Algebra, Lloyd N. Trefethen and David Bau, III.
 *
 * @param[in] A Input matrix of size n x n; forced to be 10 x 10 in this
 *              implementation.
 * @param[in] u Input vector of size n; forced to be 10 in this
 *              implementation.
 * @param[in] m Number of iterations (i.e., Krylov subspace degree).
 */
void arnoldi(double **A, double *u, int m) {
  double **Q;                  // Matrix of orthonormal basis vectors
  double **H;                  // Upper Hessenberg matrix
  const int n = 10;            // Matrix dimensions; forced to be 10 x 10
  double norm_u = 0.0, norm_v; // Vector norms
  double *v;                   // Temporary vector for calculations

  // Allocate memory for matrices and vector
  Q = (double **)malloc(n * sizeof(double *));
  H = (double **)malloc((m + 1) * sizeof(double *));
  v = (double *)malloc(n * sizeof(double));

  // Check for memory allocation failure
  if (Q == NULL || H == NULL || v == NULL) {
    fprintf(stderr, "Memory allocation failed\n");
    return;
  }

  // Allocate memory for the rows of Q
  for (int i = 0; i < n; i++) {
    Q[i] = (double *)malloc((m + 1) * sizeof(double));
    if (Q[i] == NULL) {
      fprintf(stderr, "Memory allocation failed\n");
      return;
    }
  }

  // Allocate memory for rows of H
  for (int i = 0; i <= m; i++) {
    H[i] = (double *)malloc(m * sizeof(double));
    if (H[i] == NULL) {
      fprintf(stderr, "Memory allocation failed\n");
      return;
    }
  }

  // Initialise H to zero
  for (int i = 0; i <= m; i++) {
    for (int j = 0; j < m; j++) {
      H[i][j] = 0.0;
    }
  }

  // Calculate the 2-norm of u (i.e., ||u||)
  for (int i = 0; i < n; i++) {
    norm_u += u[i] * u[i];
  }
  norm_u = sqrt(norm_u);

  // Initialize q_1=u/||u|| (i.e., our first column of Q)
  for (int i = 0; i < n; i++) {
    Q[i][0] = u[i] / norm_u;
  }

  // Main Arnoldi iteration loop
  for (int k = 0; k < m; k++) {

    // Calculate v=Aq_k
    for (int i = 0; i < n; i++) {
      v[i] = 0.0;
      for (int j = 0; j < n; j++) {
        v[i] += A[i][j] * Q[j][k];
      }
    }

    // Modified Gram-Schmidt orthogonalisation
    for (int j = 0; j <= k; j++) {

      // Calculate h_{j,k}=q_j^\intercal v (i.e., the inner product)
      H[j][k] = 0.0;
      for (int i = 0; i < n; i++) {
        H[j][k] += Q[i][j] * v[i];
      }

      // v=v-h_{j,k}q_j (i.e., the orthogonalisation step)
      for (int i = 0; i < n; i++) {
        v[i] -= H[j][k] * Q[i][j];
      }
    }

    // Calculate the 2-norm of v (i.e., ||v||)
    norm_v = 0.0;
    for (int i = 0; i < n; i++) {
      norm_v += v[i] * v[i];
    }
    norm_v = sqrt(norm_v);

    // Set h_{k+1,k}=||v||
    H[k + 1][k] = norm_v;

    // Set q_{k+1}=v/||v|| (i.e., normalisation and the next column of Q)
    for (int i = 0; i < n; i++) {
      Q[i][k + 1] = v[i] / norm_v;
    }
  }

  // Print Q as per the requirements of the assignment
  for (int i = 0; i < n; i++) {
    for (int j = 0; j <= m; j++) {
      printf("%+.15f ", Q[i][j]);
    }
    printf("\n");
  }

  // Free dynamically allocated memory
  for (int i = 0; i < n; i++) {
    free(Q[i]);
  }
  for (int i = 0; i <= m; i++) {
    free(H[i]);
  }
  free(v);
  free(Q);
  free(H);
}

/**
 * @brief Main function.
 *
 * Initialises the input matrix and vector, calls the Arnoldi iteration
 * algorithm, and computes Q_9.
 *
 * @returns 0 upon successful execution.
 */
int main() {
  const int n = 10; // Matrix dimensions
  const int m = 9;  // Krylov subspace degree

  // Allocate memory for matrix A and vector u
  double **A = (double **)malloc(n * sizeof(double *));
  double *u = (double *)malloc(n * sizeof(double));

  // Check for memory allocation failure
  if (A == NULL || u == NULL) {
    fprintf(stderr, "Memory allocation failed\n");
    return 1;
  }

  // Allocate memory for rows of A
  for (int i = 0; i < n; i++) {
    A[i] = (double *)malloc(n * sizeof(double));
    if (A[i] == NULL) {
      fprintf(stderr, "Memory allocation failed\n");
      return 1;
    }
  }

  // Initisalise A with the values given in the assignment
  double A_data[10][10] = {
      {3, 8, 7, 3, 3, 7, 2, 3, 4, 8}, {5, 4, 1, 6, 9, 8, 3, 7, 1, 9},
      {3, 6, 9, 4, 8, 6, 5, 6, 6, 6}, {5, 3, 4, 7, 4, 9, 2, 3, 5, 1},
      {4, 4, 2, 1, 7, 4, 2, 2, 4, 5}, {4, 2, 8, 6, 6, 5, 2, 1, 1, 2},
      {2, 8, 9, 5, 2, 9, 4, 7, 3, 3}, {9, 3, 2, 2, 7, 3, 4, 8, 7, 7},
      {9, 1, 9, 3, 3, 1, 2, 7, 7, 1}, {9, 3, 2, 2, 6, 4, 4, 7, 3, 5}};
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      A[i][j] = A_data[i][j];
    }
  }

  // Initialise u with the values given in the assignment
  double u_data[10] = {0.757516242460009,  2.734057963614329,
                       -0.555605907443403, 1.144284746786790,
                       0.645280108318073,  -0.085488474462339,
                       -0.623679022063185, -0.465240896342741,
                       2.382909057772335,  -0.120465395885881};
  for (int i = 0; i < n; i++) {
    u[i] = u_data[i];
  }

  // Run the Arnoldi iteration algorithm
  arnoldi(A, u, m);

  // Free allocated memory
  for (int i = 0; i < n; i++) {
    free(A[i]);
  }
  free(A);
  free(u);

  return 0;
}
