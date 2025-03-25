/**
 * @file algorithm-6.1.c
 *
 * @brief Explain briefly.
 *
 * Further explanation, if required.
 *
 * @author Ion Lipsiuc
 * @date 2025-03-25
 * @version 1.0
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

double **V_global = NULL;
double **H_global = NULL;

/**
 * @brief Explain briefly.
 *
 * Further explanation, if required.
 *
 * @param[] a Explain briefly.
 * @param[] b Explain briefly.
 * @param[] n Explain briefly.
 *
 * @returns Explain briefly.
 */
double dot_product(double *a, double *b, int n) {
  double sum = 0.0;
  for (int i = 0; i < n; i++) {
    sum += a[i] * b[i];
  }
  return sum;
}

/**
 * @brief Explain briefly.
 *
 * Further explanation, if required.
 *
 * @param[] v Explain briefly.
 * @param[] n Explain briefly.
 *
 * @returns Explain briefly.
 */
double norm(double *v, int n) { return sqrt(dot_product(v, v, n)); }

/**
 * @brief Explain briefly.
 *
 * Further explanation, if required.
 *
 * @param[] A Explain briefly.
 * @param[] v Explain briefly.
 * @param[] result Explain briefly.
 * @param[] n Explain briefly.
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
 * @brief Explain briefly.
 *
 * Further explanation, if required.
 *
 * @param[] A Explain briefly.
 * @param[] v1 Explain briefly.
 * @param[] m Explain briefly.
 *
 * @returns Explain briefly.
 */
void arnoldi(double **A, double *v1, int m) {
  int n = 10;             // Matrix size
  if (V_global != NULL) { // Free memory if it was allocated before
    for (int i = 0; i < n; i++) {
      free(V_global[i]);
    }
    free(V_global);
  }
  if (H_global != NULL) { // Free memory if it was allocated before
    for (int i = 0; i <= m; i++) {
      free(H_global[i]);
    }
    free(H_global);
  }

  // Allocate memory for orthonormal basis matrix and Hessenberg matrix
  V_global = (double **)malloc(n * sizeof(double *));
  for (int i = 0; i < n; i++) {
    V_global[i] = (double *)calloc(m + 1, sizeof(double));
  }
  H_global = (double **)malloc((m + 1) * sizeof(double *));
  for (int i = 0; i <= m; i++) {
    H_global[i] = (double *)calloc(m, sizeof(double));
  }

  // Allocate memory for intermediate results as per the pseudocode
  double *w = (double *)malloc(n * sizeof(double));

  // 1. Choose a vector v_1 of norm 1
  double v1_norm = norm(v1, n);
  for (int i = 0; i < n; i++) {
    V_global[i][0] = v1[i] / v1_norm;
  }

  // 2. For j = 1, 2, ..., m Do:
  for (int j = 0; j < m; j++) {

    // Allocate memory for v_j
    double *v_j = (double *)malloc(n * sizeof(double));
    for (int i = 0; i < n; i++) {
      v_j[i] = V_global[i][j];
    }

    // Compute A * v_j
    mvm(A, v_j, w, n);

    // 3. Compute h_{i, j} = (A * v_j, v_i) for i = 1, 2, ..., j
    for (int i = 0; i <= j; i++) {

      // Allocate memory for v_i
      double *v_i = (double *)malloc(n * sizeof(double));
      for (int k = 0; k < n; k++) {
        v_i[k] = V_global[k][i];
      }

      // Compute (A * v_j, v_i)
      H_global[i][j] = dot_product(w, v_i, n);

      // 4. Compute w_j := A * v_j - sum_{i = 1}^j h_{i, j} * v_i
      for (int k = 0; k < n; k++) {
        w[k] -= H_global[i][j] * v_i[k];
      }

      free(v_i);
    }

    // 5. h_{j + 1, j} = ||w_j||_2
    H_global[j + 1][j] = norm(w, n);

    // 6. If h_{j + 1, j} = 0 then Stop
    if (fabs(H_global[j + 1][j]) < 1e-10) {
      printf("Division by zero is not possible");
      free(v_j);
      break;
    }

    // 7. v_{j + 1} = w_j / h_{j + 1, j}$
    for (int i = 0; i < n; i++) {
      V_global[i][j + 1] = w[i] / H_global[j + 1][j];
    }

    free(v_j);
  }

  free(w);
}

/**
 * @brief Main function.
 *
 * Further explanation, if required.
 *
 * @returns 0 upon successful execution.
 */
int main() {
  int n = 10; // Matrix size
  int m = 9;

  // Allocate memory for input matrix
  double **A = (double **)malloc(n * sizeof(double *));
  for (int i = 0; i < n; i++) {
    A[i] = (double *)malloc(n * sizeof(double));
  }

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
      printf("%10.6f ", V_global[i][j]);
    }
    printf("\n");
  }
  printf("\n");

  // Print the entire Q matrix which is stored in V_global
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < m; j++) {
      printf("%10.6f ", H_global[i][j]);
    }
    printf("\n");
  }

  // Free memory
  for (int i = 0; i < n; i++) { // Input and orthonormal basis matrices
    free(A[i]);
    free(V_global[i]);
  }
  free(A);        // Input matrix
  free(V_global); // Orthonormal basis matrix
  for (int i = 0; i <= m; i++) {
    free(H_global[i]); // Hessenberg matrix
  }
  free(H_global); // Hessenberg matrix
  free(v1);       // Input vector

  return 0;
}
