# Case Studies in High-Performance Computing

## Assignment 1 - Kyrlov Subspace Methods and GMRES

### Mathematical Background

#### Notation and Linear System Representation

The generalised minimal residual (GMRES) algorithm is an iterative method for solving the linear system:

```math
Ax=b,
```

where:

- $A$ is an arbitrary $n\times n$ non-singular matrix.
- $x$ is the unknown solution vector.
- $b$ is a given right-hand side vector.

GMRES belongs to the class of Krylov subspace methods, which build approximate solutions within a sequence of nested subspaces. The method works by iteratively constructing an orthonormal basis for the Krylov subspace and solving a least-squares problem at each step.

#### Krylov Subspaces

The Krylov subspace of dimension $m$ associated with $A$ and $b$ is defined as:

```math
\mathcal{K}_m(A,b)=\text{span}\{b,Ab,A^2b,\ldots,A^{m-1}b\}.
```

This subspace contains all possible linear combinations of matrix powers applied to the initial residual $r_0=b-Ax_0$. Some properties which are important to remember are as follows:

- The dimension of $\mathcal{K}_m(A,b)$ is at most $n$.
- Krylov methods approximate the solution by projecting it onto these subspaces and enforcing optimality conditions.

#### Arnoldi Iteration

The algorithm we employ comes from Iterative Methods for Sparse Linear Systems, 2nd Ed., Yousef Saad. It uses the modified Gram-Schmidt algorithm instead of the standard Gram-Schmidt algorithm as it is much more reliable in the presence of round-off.

Given a matrix $A\in\mathbf{R}^{n\times n}$ and an initial unit vector $v_1$, the Arnoldi iteration constructs an orthonormal basis $`\{v_1,v_2,\ldots,v_m\}`$ for the Krylov subspace $\mathcal{K}_m(A,v_1)$ along with an upper Hessenberg matrix $H_m$. The pseudocode is as follows:

1. **Initialise First Basis Vector**:

    - Initialise the first basis vector where $b$ is the initial residual or given starting vector:

```math
v_1=\frac{b}{\|b\|_2}.
```

2. **Arnoldi Iteration (i.e., for $j=1,\ldots,m$)**:

    - **Matrix-Vector Multiplication**: $w_j=Av_j$ (i.e., a candidate vector for the new basis vector).
    - **Modified Gram-Schmidt Orthogonalisation**: For each previously computed $v_i$ (where $i=1,\ldots,j$), compute the projection coefficient $h_{i,j}=\langle w_j,v_i\rangle$ and subtract the projection from $w_j$ to enforce orthogonality:

```math
w_j=w_j-h_{i,j}v_i.
```

    - **Compute Norm of the Remaining Vector**: $`h_{j+1,j}=\|w_j\|_2`$ and if $h_{j+1,j}=0$, then stop the process.
    - **Normalise the New Basis Vector**: This is obtained by carrying out the following:

```math
v_{j+1}=\frac{w_j}{h_{j+1,j}}.
```

The above the results in the relation:

```math
AQ_m=Q_{m+1}H_m,
```
where:

- $Q_m=[v_1\ v_2\ \ldots\ v_m]$ is an orthonormal basis of the Krylov subspace.
- $H_m$ is an $(m+1)\times m$ upper Hessenberg matrix containing the computed $h_{i,j}$ coefficients.

#### Generalised Minimal Residual Algorithm

GMRES seeks the approximate solution $x_m$ in the Krylov subspace by minimising the residual norm:

```math
\|Ax_m-b\|_2=\min_{x\in\mathcal{K}_m(A,b)}\|Ax-b\|_2.
```

Our implementation consists of two mains steps:

1. **Arnoldi Iteration**: Generate an orthonormal basis for the Krylov subspace.
2. **Least-Squares Solution Using Givens Rotations**: Solve the least-squares problem to determine the best approximation for $x_m$.

The pseudocode is as follows:

1. **Initialise First Basis Vector**:

    - Compute the initial residual $r_0=b-Ax_0$.
    - Compute its Euclidean norm $`\beta=\|r_0\|_2`$.
    - Normalise the first basis vector $v_1=\frac{r_0}{\beta}$.

2. **Arnoldi Iteration (i.e., for $j=1,\ldots,m$)**:

    - **Matrix-Vector Multiplication**: $w_j=Av_j$ (i.e., a candidate vector for the new basis vector).
    - **Modified Gram-Schmidt Orthogonalisation**: For each previously computed $v_i$ (where $i=1,\ldots,j$), compute the projection coefficient $h_{i,j}=\langle w_j,v_i\rangle$ and subtract the projection from $w_j$ to enforce orthogonality:

```math
w_j=w_j-h_{i,j}v_i.
```

    - **Compute Norm of the Remaining Vector**: $`h_{j+1,j}=\|w_j\|_2`$ and if $h_{j+1,j}=0$, then stop the process.
    - **Normalise the New Basis Vector**: This is obtained by carrying out the following:

```math
v_{j+1}=\frac{w_j}{h_{j+1,j}}.
```

> Note that the above step is an exact copy of the Arnoldi algorithm. As such, carrying out the above results in the same relation we have seen before (i.e., $AQ_m=Q_{m+1}H_m$).

3. **Apply Givens Rotations to Transform $H_m$ into an Upper Triangular System**:

    - Define the Givens rotation matrices $\Omega_i$ to eliminate the subdiagonal entries of $H_m$. Each rotation is defined as

```math
\Omega_i=\begin{bmatrix}1&&&&&\\&\ddots&&&&\\&&c_i&s_i&&\\&&-s_i&c_i&&\\&&&&\ddots&\\&&&&&1\end{bmatrix},$$ where: $$c_i=\frac{h_{i,i}}{\sqrt{h_{i,i}^2+h{i+1,i}^2}},\ s_i=\frac{h_{i+1,i}}{\sqrt{h_{i,i}^2+h{i+1,i}^2}}.
```

    - Apply each $\Omega_i$ to both $H_m$ and the transformed right-hand side vector $g$ to obtain and upper triangular system.

4. **Solve the Least-Squares Problem**:

    - The least-squares system is $`\min_y\|\beta e_1-H_my\|_2`$ which, after applying Givens rotations, turns into solving the triangular system $R_my_m=g_m$ using back substitution, where $R_m$ is the transformed $H_m$ after applying all Givens rotations.

5. **Compute the Final Solution Approximation**:

    - Final solution is computed using $x_m=x_0+Q_my_m$.
