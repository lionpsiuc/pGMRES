/**
 * @file arnoldi.c
 *
 * @brief Implementation of Arnoldi's algorithm, as per the pseudocode given in
 *        Algorithm 6.2: Arnoldi-Modified Gram-Schmidt, within Iterative Methods
 *        for Sparse Linear Systems, 2nd Ed., Yousef Saad
 *
 * @author Ion Lipsiuc
 * @date 2025-03-25
 * @version 1.0
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

double **H_global = NULL;
double **V_global = NULL;

/**
 * @brief Allocates memory for a matrix.
 *
 * @param[in] rows Number of rows.
 * @param[in] cols Number of columns.
 *
 * @returns Pointer to the allocated matrix.
 */
double **allocate_matrix(int rows, int cols) {
  double **M = (double **)malloc(rows * sizeof(double *));
  for (int i = 0; i < rows; i++) {
    M[i] = (double *)calloc(cols, sizeof(double));
  }
  return M;
}

/**
 * @brief Frees memory allocated for a matrix.
 *
 * @param[in,out] M Pointer to the matrix.
 * @param[in] rows Number of rows in the matrix.
 */
void free_matrix(double **M, int rows) {
  for (int i = 0; i < rows; i++) {
    free(M[i]);
  }
  free(M);
}

/**
 * @brief Computes the dot product of two vectors.
 *
 * @param[in] a First vector.
 * @param[in] b Second vector.
 * @param[in] n Size of the vectors.
 *
 * @returns The computed dot product.
 */
double dot_product(double *a, double *b, int n) {
  double sum = 0.0;
  for (int i = 0; i < n; i++) {
    sum += a[i] * b[i];
  }
  return sum;
}

/**
 * @brief Computes the Euclidean norm of a vector.
 *
 * @param[in] v Vector whose norm is computed.
 * @param[in] n Size of the vector.
 *
 * @returns The computed norm.
 */
double norm(double *v, int n) { return sqrt(dot_product(v, v, n)); }

/**
 * @brief Performs matrix-vector multiplication.
 *
 * @param[in] A Input square matrix.
 * @param[in] v Input vector.
 * @param[out] result Output vector storing the result of A * v.
 * @param[in] n Size of the matrix and vectors.
 */
void mvm(double **A, double *v, double *result, int n) {
  for (int i = 0; i < n; i++) {
    result[i] = 0.0;
    for (int j = 0; j < n; j++) {
      result[i] += A[i][j] * v[j];
    }
  }
}

/**
 * @brief Arnoldi iteration to generate an orthonormal basis for the Krylov
 *        subspace.
 *
 * The Arnoldi iteration constructs an orthonormal basis for the Krylov subspace
 * and produces an upper Hessenberg matrix.
 *
 * @param[in] A Input square matrix.
 * @param[in] v1 Initial vector.
 * @param[in] m Number of Arnoldi iterations.
 */
void arnoldi(double **A, double *v1, int m) {
  int n = 10;             // Matrix size
  if (H_global != NULL) { // Free memory if it was allocated before
    free_matrix(H_global, m);
  }
  if (V_global != NULL) { // Free memory if it was allocated before
    free_matrix(V_global, n);
  }

  // Allocate memory for orthonormal basis and Hessenberg matrices
  H_global = allocate_matrix(m + 1, m);
  V_global = allocate_matrix(n, m + 1);

  // Allocate memory for intermediate results as per the pseudocode
  double *w = (double *)malloc(n * sizeof(double));

  double v1_norm = norm(v1, n);
  for (int i = 0; i < n; i++) {
    V_global[i][0] = v1[i] / v1_norm; // 1. Choose a vector v_1 of norm 1
  }

  // 2. For j = 1, 2, ..., m Do:
  for (int j = 0; j < m; j++) {

    // Allocate memory for v_j and initialise it
    double *v_j = (double *)malloc(n * sizeof(double));
    for (int i = 0; i < n; i++) {
      v_j[i] = V_global[i][j];
    }

    mvm(A, v_j, w, n); // 3. Compute w_j := A * v_j
    free(v_j);         // Free memory for v_j

    // 4. For i = 1, ..., j Do:
    for (int i = 0; i <= j; i++) {

      // Allocate memory for v_i and initialise it
      double *v_i = (double *)malloc(n * sizeof(double));
      for (int k = 0; k < n; k++) {
        v_i[k] = V_global[k][i];
      }

      H_global[i][j] =
          dot_product(w, v_i, n); // 5. Compute h_{i, j} = (w_j, v_i)

      // 6. w_j := w_j - h_{i, j} * v_i
      for (int k = 0; k < n; k++) {
        w[k] -= H_global[i][j] * v_i[k];
      }

      free(v_i);
    }

    H_global[j + 1][j] = norm(w, n); // 8. h_{j + 1, j} = ||w_j||_2

    // 8: If h_{j + 1, j} = 0 Stop
    if (fabs(H_global[j + 1][j]) < 1e-10) {
      printf("Division by zero is not possible\n");
      break;
    }

    // 9. v_{j + 1} = w_j / h_{j + 1, j}
    for (int i = 0; i < n; i++) {
      V_global[i][j + 1] = w[i] / H_global[j + 1][j];
    }
  }

  free(w);
}

/**
 * @brief Main function.
 *
 * Initialises the matrix and vector, and then runs Arnoldi's algorithm.
 *
 * @returns 0 upon successful execution.
 */
int main() {
  int n = 10; // Matrix size
  int m = 9;

  // Allocate memory for input matrix
  double **A = allocate_matrix(n, n);

  // Initialise input matrix
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

  // Allocate memory for input vector (i.e., b, as is given in the
  // assignment, which is the first column of the Q matrix)
  double *v1 = (double *)malloc(n * sizeof(double));

  // Initialise input vector
  double v1_data[10] = {0.757516242460009,  2.734057963614329,
                        -0.555605907443403, 1.144284746786790,
                        0.645280108318073,  -0.085488474462339,
                        -0.623679022063185, -0.465240896342741,
                        2.382909057772335,  -0.120465395885881};
  for (int i = 0; i < n; i++) {
    v1[i] = v1_data[i];
  }

  // Run Arnoldi
  arnoldi(A, v1, m);

  // Print the entire Q matrix which is stored in V_global
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < m + 1; j++) {
      printf("%9.5f ", V_global[i][j]);
    }
    printf("\n");
  }

  printf("\n");

  // Print the Hessenberg matrix
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < m; j++) {
      printf("%9.5f ", H_global[i][j]);
    }
    printf("\n");
  }

  // Free memory
  free_matrix(A, n);        // Input matrix
  free_matrix(V_global, n); // Orthonormal basis matrix
  free_matrix(H_global, m); // Hessenberg matrix
  free(v1);                 // Input vector

  return 0;
}
