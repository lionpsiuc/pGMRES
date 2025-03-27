/**
 * @file gmres-mpi.c
 *
 * @brief Implementation of a parallelised GMRES algorithm using MPI, as per
 *        the pseudocode given in Algorithm 6.9: GMRES, within Iterative Methods
 *        for Sparse Linear Systems, 2nd Ed., Yousef Saad, using 6.5.3 Practical
 *        Implementation Issues as a reference for solving the least-squares
 *        problem.
 *
 * @author Ion Lipsiuc
 * @date 2025-03-26
 * @version 1.0
 */

#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

// Global MPI variables
int my_rank; // Rank of this process
int np;      // Number of processes
MPI_Comm comm = MPI_COMM_WORLD;

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
 * @brief Computes the parallel dot product of two vectors.
 *
 * @param[in] a First local vector.
 * @param[in] b Second local vector.
 * @param[in] local_n Size of the local vectors.
 *
 * @returns The computed global dot product.
 */
double dot_product(double *a, double *b, int local_n) {
  double local_sum = 0.0;
  double global_sum = 0.0;

  // Compute local sum
  for (int i = 0; i < local_n; i++) {
    local_sum += a[i] * b[i];
  }

  // Sum up results from all processes
  MPI_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, comm);

  return global_sum;
}

/**
 * @brief Computes the Euclidean norm of a vector in parallel.
 *
 * @param[in] v Local vector whose norm is computed.
 * @param[in] local_n Size of the local vector.
 *
 * @returns The computed global norm.
 */
double norm(double *v, int local_n) { return sqrt(dot_product(v, v, local_n)); }

/**
 * @brief Performs parallel matrix-vector multiplication.
 *
 * @param[in] v Input local vector.
 * @param[out] result Output local vector.
 * @param[in] local_n Size of the local part.
 * @param[in] n Global size.
 */
void mvm(double *v, double *result, int local_n, int n) {

  // Calculate global row offset for this process
  int row_offset = my_rank * local_n;

  // Exchange boundary elements with neighbouring processes
  double send_up = 0.0, send_down = 0.0;
  double recv_up = 0.0, recv_down = 0.0;
  MPI_Status status;

  // Send up and receive from below
  if (my_rank < np - 1) {
    send_down = v[local_n - 1];
    MPI_Sendrecv(&send_down, 1, MPI_DOUBLE, my_rank + 1, 0, &recv_down, 1,
                 MPI_DOUBLE, my_rank + 1, 1, comm, &status);
  }

  // Send down and receive from above
  if (my_rank > 0) {
    send_up = v[0];
    MPI_Sendrecv(&send_up, 1, MPI_DOUBLE, my_rank - 1, 1, &recv_up, 1,
                 MPI_DOUBLE, my_rank - 1, 0, comm, &status);
  }

  // Compute local matrix-vector product
  for (int i = 0; i < local_n; i++) {
    int global_i = row_offset + i;
    result[i] = -4.0 * v[i]; // Diagonal element

    // Lower off-diagonal
    if (i > 0) {
      result[i] += v[i - 1];
    } else if (my_rank > 0) {
      result[i] += recv_up;
    }

    // Upper off-diagonal
    if (i < local_n - 1) {
      result[i] += v[i + 1];
    } else if (my_rank < np - 1) {
      result[i] += recv_down;
    }
  }
}

/**
 * @brief Runs the GMRES algorithm to solve a linear system A * x = b.
 *
 * This function uses the Arnoldi iteration to generate a Krylov subspace and
 * solves the resulting least-squares problem using Givens rotations.
 *
 * @param[in] b Right-hand side local vector.
 * @param[in] local_n Size of the local part.
 * @param[in] n Global size of the matrix and vectors.
 * @param[in] m Maximum iterations.
 * @param[out] residual_history Array storing the residual norm at each
 *                              iteration.
 *
 * @returns The approximate solution local vector.
 */
double *gmres(double *b, int local_n, int n, int m, double *residual_history) {
  double *x =
      (double *)calloc(local_n, sizeof(double)); // x_0 = 0, for simplicity
  double *r = (double *)malloc(local_n * sizeof(double)); // r_0
  for (int i = 0; i < local_n; i++) {
    r[i] = b[i]; // 1. Compute r_0 = b - A * x_0
  }

  // 1. beta := ||r_0||_2
  double beta = norm(r, local_n);

  // Allocate memory for our matrix to store the Arnoldi basis vectors
  double **V = allocate_matrix(local_n, m + 1);

  for (int i = 0; i < local_n; i++) {
    V[i][0] = r[i] / beta; // 1. v_1 := r_0 / beta
  }
  free(r);
  double **H = allocate_matrix(m + 1, m); // The same on all processes

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
    double *v_j = (double *)malloc(local_n * sizeof(double));
    double *w = (double *)malloc(local_n * sizeof(double));
    for (int i = 0; i < local_n; i++) {
      v_j[i] = V[i][j]; // Extract v_j from V
    }
    mvm(v_j, w, local_n, n); // 4. Compute w_j := A * v_j
    free(v_j);

    // 5. For i = 1, ..., j Do:
    for (int i = 0; i <= j; i++) {
      double *v_i = (double *)malloc(local_n * sizeof(double));
      for (int k = 0; k < local_n; k++) {
        v_i[k] = V[k][i];
      }
      H[i][j] = dot_product(w, v_i, local_n); // 6. h_{i, j} = (w_j, v_i
      for (int k = 0; k < local_n; k++) {
        w[k] -= H[i][j] * v_i[k]; // 7. w_j := w_j - h_{i, j} * v_i
      }
      free(v_i);
    }

    // 9. h_{j + 1, j} = ||w_j||_2. If h_{j + 1, j} = 0 then set m := j and go
    // to 12
    H[j + 1][j] = norm(w, local_n);
    if (fabs(H[j + 1][j]) < 1e-14) {
      double current_res = fabs(g[j + 1]); // Fill remainder of the residual
                                           // history with the current residual
      for (int k = j; k < m; k++) {
        residual_history[k] = current_res;
      }
      free(w);
      break;
    }

    // 10. v_{j + 1} = w_j / h_{j + 1, j}
    for (int i = 0; i < local_n; i++) {
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
    H[j][j] = c[j] * H[j][j] + s[j] * H[j + 1][j];
    H[j + 1][j] = 0.0;

    // Update g
    double temp_g = c[j] * g[j];
    g[j + 1] = -s[j] * g[j];
    g[j] = temp_g;

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
  for (int i = 0; i < local_n; i++) {
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
  free_matrix(V, local_n);

  return x;
}

/**
 * @brief Main function.
 *
 * Run GMRES on test cases and save residuals.
 *
 * @returns 0 upon successful execution.
 */
int main(int argc, char *argv[]) {

  // Initialise MPI
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(comm, &my_rank);
  MPI_Comm_size(comm, &np);

  int sizes[] = {8, 16, 32, 64, 128, 256};
  int num_sizes = sizeof(sizes) / sizeof(sizes[0]);
  FILE *fp = NULL;

  // Open file for writing residuals on root process
  if (my_rank == 0) {
    fp = fopen("gmres-mpi-residuals.csv", "w");
    fprintf(fp, "n,iteration,residual\n");
  }

  for (int idx = 0; idx < num_sizes; idx++) {
    int n = sizes[idx];
    int m = n / 2;

    // Calculate local matrix size
    int local_n = n / np;

    // Ensure matrix size is divisible by number of processes
    if (n % np != 0) {
      if (my_rank == 0) {
        printf("Matrix size %d not divisible by processes %d\n", n, np);
      }
      MPI_Finalize();
      return 1;
    }

    // Initialise local part of the right-hand side vector
    double *b = (double *)malloc(local_n * sizeof(double));
    int start_idx = my_rank * local_n;
    for (int i = 0; i < local_n; i++) {
      int global_i = start_idx + i;
      if (global_i < n - 1) {
        b[i] = (global_i + 1) / (double)n;
      } else {
        b[i] = 1.0;
      }
    }

    // Allocate array to store residuals at each iteration
    double *res_history = (double *)calloc(m, sizeof(double));

    // Run GMRES
    double *x = gmres(b, local_n, n, m, res_history);

    double norm_b = norm(b, local_n); // For normalising residuals

    // Gather and print residuals on root process
    if (my_rank == 0) {
      for (int j = 0; j < m; j++) {
        double normalised_res = res_history[j] / norm_b;
        fprintf(fp, "%d,%d,%.15e\n", n, j + 1, normalised_res);
      }
    }

    // Gather the complete solution on the root process
    double *global_x = NULL;
    if (my_rank == 0) {
      global_x = (double *)malloc(n * sizeof(double));
    }

    // Gather all local solutions on the root process
    MPI_Gather(x, local_n, MPI_DOUBLE, global_x, local_n, MPI_DOUBLE, 0, comm);

    if (my_rank == 0) {

      // Print first five and last five values of solution vector
      printf("For n = %d:\n", n);
      for (int i = 0; i < 2; i++) {
        printf("%.15e ", global_x[i]);
      }
      printf("... ");
      for (int i = n - 2; i < n; i++) {
        printf("%.15e ", global_x[i]);
      }
      printf("\n\n");

      free(global_x);
    }

    // Clean up
    free(b);
    free(res_history);
    free(x);
  }
  if (my_rank == 0) {
    fclose(fp);
    printf("Wrote residual history to file\n");
  }

  // Finalise MPI
  MPI_Finalize();
  return 0;
}
