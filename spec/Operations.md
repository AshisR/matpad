| Function | Operator | Description                                                                              |
| add      | +        | Add two or more matrices                                                                 |
| sub      | -        | Subtract one matrix from another. Unary `-expr` also negates supported expressions.      |
| eq       | ==       | Check whether two matrices have the same shape and equal entries. Returns a boolean.     |
| mult     | *        | Multiply two or more matrices                                                            |
| pow      | ^        | Power of a matrix                                                                        |
| det      |          | Determinant of a matrix                                                                  |
| tr       |          | Trace of a determinant                                                                   |
| T        |          | Transpose of a matrix                                                                    |
| ref      |          | Convert a matrix to row echelon form                                                     |
| rref     |          | Convert a matrix to reduced row echelon form                                             |
| dist     |          | Computes distance between two same-shaped vectors or matrices. For matrices this is the Frobenius distance, equivalent to `sqrt(tr((A-B)^T (A-B)))`. |
| angle    |          | Computes the angle in radians between two vectors                                        |
| dot      |          | Dot product of two matrices                                                              |
| qr       |          | QR factorization of a matrix. Should return both the matrices as output.                 |
| diag     |          | Diagonalization of a matrix                                                              |
| solve    |          | Solves a linear matrix equation, or system of linear scalar equations of the form Ax = b |
| inv      |          | Computes the multiplicative inverse of a matrix                                          |
| rank     |          | Computes the rank of a matrix                                                            |
| isIdentity |        | Returns true when the input is an identity matrix; otherwise returns false               |
| isDiagonal |        | Returns true when all off-diagonal entries are zero                                      |
| isSymmetric |       | Returns true when the matrix is equal to its transpose                                   |
| isUpperTriangular | | Returns true when all entries below the main diagonal are zero                         |
| isOrthogonal |      | Returns true when the matrix is orthogonal, i.e., `A^T A = I`                           |
| isOrthonormal |     | Returns true when the columns of the matrix are orthonormal                              |
| isIndependent |     | Returns true when the matrix columns are linearly independent                             |
| lstsq    |          | Returns the least-squares solution to a linear matrix equation                           |
| eig      |          | Computes the eigenvalues and right eigenvectors of a general square array.               |
| schur    |          | Computes a Schur decomposition of a square matrix and returns unitary and triangular factors. |
| jnf      |          | Computes Jordan normal form of a square matrix and returns transform and Jordan factors. |
| norm     |          | Computes a matrix or vector norm                                                         |
| svd      |          | Singular Value Decomposition — returns U, S (diagonal singular-value matrix), and Vt    |
| gs       |          | Gram-Schmidt orthogonalisation (modified algorithm) — returns Q (m × n) whose columns are an orthonormal basis for the column space of A. Requires linearly independent columns. |
| I        |          | Identity matrix — `I(n)` returns the n×n identity matrix. |
| elem_scale |        | Scale elementary matrix — `elem_scale(n, p, i)`: n×n identity with row i multiplied by scalar p. Row index is 1-based. |
| elem_swap |         | Swap elementary matrix — `elem_swap(n, i, j)`: n×n identity with rows i and j exchanged. Row indices are 1-based; i ≠ j. |
| elem_shear |        | Shear elementary matrix — `elem_shear(n, p, i, j)`: n×n identity with E[i,j] = p; left-multiplying adds p·row j to row i. Row indices are 1-based; i ≠ j. |
| lu       |          | LU decomposition with partial pivoting — returns permutation matrix P, unit lower-triangular matrix L, and upper-triangular matrix U such that A = P·L·U. Works for any matrix shape. |
| isSimilar |         | Returns true when two square matrices A and B are similar, i.e., there exists an invertible matrix P such that B = P⁻¹AP. Similarity is checked via Jordan Normal Form (requires SymPy). Raises an error for non-square inputs. |
