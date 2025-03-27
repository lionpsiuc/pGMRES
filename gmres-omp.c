/**
 * @file gmres-omp.c
 *
 * @brief Implementation of a parallelised GMRES algorithm using OpenMP, as per
 *        the pseudocode given in Algorithm 6.9: GMRES, within Iterative Methods
 *        for Sparse Linear Systems, 2nd Ed., Yousef Saad, using 6.5.3 Practical
 *        Implementation Issues as a reference for solving the least-squares
 *        problem.
 *
 * @author Ion Lipsiuc
 * @date 2025-03-25
 * @version 1.0
 */

#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

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
#pragma omp parallel for reduction(+ : sum)
  for (int i = 0; i < n; i++)
    sum += a[i] * b[i];
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
#pragma omp parallel for
  for (int i = 0; i < n; i++) {
    result[i] = 0.0;
    for (int j = 0; j < n; j++) {
      result[i] += A[i][j] * v[j];
    }
  }
}

/**
 * @brief Runs the GMRES algorithm to solve a linear system A * x = b.
 *
 * This function uses the Arnoldi iteration to generate a Krylov subspace and
 * solves the resulting least-squares problem using Givens rotations.
 *
 * @param[in] A Input square matrix.
 * @param[in] b Right-hand side vector.
 * @param[in] n Size of the matrix and vectors.
 * @param[in] m Maximum iterations.
 * @param[out] residual_history Array storing the residual norm at each
 *                              iteration.
 *
 * @returns The approximate solution vector.
 */
double *gmres(double **A, double *b, int n, int m, double *residual_history) {
  double *x = (double *)calloc(n, sizeof(double));  // x_0 = 0, for simplicity
  double *r = (double *)malloc(n * sizeof(double)); // r_0
#pragma omp parallel for
  for (int i = 0; i < n; i++) {
    r[i] = b[i]; // 1. Compute r_0 = b - A * x_0
  }

  // 1. beta := ||r_0||_2
  double beta = norm(r, n);

  // Allocate memory for our matrix to store the Arnoldi basis vectors
  double **V = allocate_matrix(n, m + 1);

#pragma omp parallel for
  for (int i = 0; i < n; i++) {
    V[i][0] = r[i] / beta; // 1. v_1 := r_0 / beta
  }
  free(r);
  double **H = allocate_matrix(m + 1, m);

  // Define the Givens rotation matrices (here, we only need to store scalars
  // since it is enough) as described in 6.5.3 Practical Implementation Issues
  // of Iterative Methods for Sparse Linear Systems, 2nd Ed., Yousef Saad
  double *c = (double *)calloc(m, sizeof(double));
  double *s = (double *)calloc(m, sizeof(double));

  // This is the g vector as in 6.5.3 Practical Implementation Issues
  double *g = (double *)calloc(m + 1, sizeof(double));
  g[0] = beta; // First entry is just beta

  // 3. For j = 1, 2, ..., m Do:
  int j; // Needed for recording how many iterations we have completed
  for (j = 0; j < m; j++) {
    double *v_j = (double *)malloc(n * sizeof(double));
    double *w = (double *)malloc(n * sizeof(double));
#pragma omp parallel for
    for (int i = 0; i < n; i++) {
      v_j[i] = V[i][j]; // Extract v_j from V
    }
    mvm(A, v_j, w, n); // 4. Compute w_j := A * v_j
    free(v_j);

    // 5. For i = 1, ..., j Do:
    for (int i = 0; i <= j; i++) {
      double *v_i = (double *)malloc(n * sizeof(double));
#pragma omp parallel for
      for (int k = 0; k < n; k++) {
        v_i[k] = V[k][i];
      }
      H[i][j] = dot_product(w, v_i, n); // 6. h_{i, j} = (w_j, v_i)
#pragma omp parallel for
      for (int k = 0; k < n; k++) {
        w[k] -= H[i][j] * v_i[k]; // 7. w_j := w_j - h_{i, j} * v_i
      }
      free(v_i);
    }

    // 9. h_{j + 1, j} = ||w_j||_2. If h_{j + 1, j} = 0 then set m := j and go
    // to 12
    H[j + 1][j] = norm(w, n);
    if (fabs(H[j + 1][j]) < 1e-14) {
      double current_res = fabs(g[j + 1]); // Fill remainder of the residual
                                           // history with the current residual
#pragma omp parallel for
      for (int k = j; k < m; k++) {
        residual_history[k] = current_res;
      }
      free(w);
      break;
    }

    // 10. v_{j + 1} = w_j / h_{j + 1, j}
#pragma omp parallel for
    for (int i = 0; i < n; i++) {
      V[i][j + 1] = w[i] / H[j + 1][j];
    }

    free(w);

    // Apply previously computed rotations
    for (int i = 0; i < j; i++) {
      double temp = c[i] * H[i][j] + s[i] * H[i + 1][j];
      H[i + 1][j] = -s[i] * H[i][j] + c[i] * H[i + 1][j];
      H[i][j] = temp;
    }

    // Compute new Givens rotation
    double denom = sqrt(H[j][j] * H[j][j] + H[j + 1][j] * H[j + 1][j]);
    c[j] = H[j][j] / denom;
    s[j] = H[j + 1][j] / denom;

    // Apply the rotation to our Hessenberg matrix
    double temp2 = c[j] * g[j] + s[j] * 0.0;
    double temp3 = -s[j] * g[j] + c[j] * 0.0;
    H[j][j] = c[j] * H[j][j] + s[j] * H[j + 1][j];
    H[j + 1][j] = 0.0;

    // Update g
    g[j + 1] = -s[j] * g[j];
    g[j] = temp2;

    // Record residual norm
    residual_history[j] = fabs(g[j + 1]);
  }

  int used_iters = (j < m) ? j + 1 : m;

  // Solve the least-squares problem
  double *y = (double *)calloc(used_iters, sizeof(double));
  for (int i = used_iters - 1; i >= 0; i--) {
    double sum = g[i];
    for (int k = i + 1; k < used_iters; k++) {
      sum -= H[i][k] * y[k];
    }
    y[i] = sum / H[i][i]; // 12. Compute y_m the minimizer of ||beta * e_1 -
                          // bar{H}_m * y||_2 = ||bar{g}_m - bar{R}_m * y||_2
  }

  // 12. x_m = x_0 + V_m * y_m
#pragma omp parallel for
  for (int i = 0; i < n; i++) {
    for (int col = 0; col < used_iters; col++) {
      x[i] += V[i][col] * y[col];
    }
  }

  // Free allocated resources
  free(c);
  free(g);
  free(s);
  free(y);
  free_matrix(H, m + 1);
  free_matrix(V, n);

  return x;
}

/**
 * @brief Main function.
 *
 * Run GMRES on test cases and save residuals.
 *
 * @returns 0 upon successful execution.
 */
int main() {
  int sizes[] = {8, 16, 32, 64, 128, 256};
  int num_sizes = sizeof(sizes) / sizeof(sizes[0]);
  FILE *fp = fopen("gmres-omp-residuals.csv", "w");
  fprintf(fp, "n,iteration,residual\n");

  // Set number of threads
  int num_threads = 16;
  omp_set_num_threads(num_threads);

  for (int idx = 0; idx < num_sizes; idx++) {
    int n = sizes[idx];
    int m = n / 2;

    // Allocate and initialise input matrix
    double **A = allocate_matrix(n, n);
#pragma omp parallel for
    for (int i = 0; i < n; i++) {
      A[i][i] = -4.0;
      if (i + 1 < n)
        A[i][i + 1] = 1.0;
      if (i - 1 >= 0)
        A[i][i - 1] = 1.0;
    }

    // Allocate and initialise input vector
    double *b = (double *)malloc(n * sizeof(double));
#pragma omp parallel for
    for (int i = 0; i < n - 1; i++) {
      b[i] = (i + 1) / (double)n;
    }
    b[n - 1] = 1.0;

    // Allocate array to store residuals at each iteration
    double *res_history = (double *)calloc(m, sizeof(double));

    // Run GMRES
    double *x = gmres(A, b, n, m, res_history);

    // Print out all residuals, which are normalised
    double norm_b = 0.0;
#pragma omp parallel for reduction(+ : norm_b)
    for (int i = 0; i < n; i++) {
      norm_b += b[i] * b[i];
    }
    norm_b = sqrt(norm_b);
    for (int j = 0; j < m; j++) {
      double normalised_res = res_history[j] / norm_b;
      fprintf(fp, "%d,%d,%.15e\n", n, j + 1, normalised_res);
    }

    // Print first five and last five values of solution vector
    printf("For n = %d:\n", n);
    for (int i = 0; i < 2; i++) {
      printf("%.15e ", x[i]);
    }
    printf("... ");
    for (int i = n - 2; i < n; i++) {
      printf("%.15e ", x[i]);
    }
    printf("\n\n");

    // Clean up
    free(b);
    free(res_history);
    free(x);
    free_matrix(A, n);
  }
  fclose(fp);
  printf("Wrote residual history to file\n");
  return 0;
}
