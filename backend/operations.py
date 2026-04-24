"""
Matrix operations catalog and NumPy-backed implementations.

Each public operation function receives a list of already-evaluated arguments
(numpy arrays or plain Python scalars) and returns a result dict:

    {"type": "scalar"|"vector"|"matrix"|"boolean"|"multi_output", "value": ..., "outputs": ...}

Multi-output operations return:
    {"type": "multi_output", "outputs": {"label": {"type": ..., "value": ...}, ...}}
"""
from __future__ import annotations

import numpy as np
from scipy import linalg as sp_linalg
from dataclasses import dataclass

# Tolerance used for all floating-point boolean predicates
FLOAT_TOL = 1e-9


# ─── Catalog ──────────────────────────────────────────────────────────────────

@dataclass
class CatalogEntry:
    name: str
    operator: str | None
    description: str
    min_args: int
    max_args: int | None  # None = variadic


CATALOG: list[CatalogEntry] = [
    CatalogEntry("add",               "+",  "Add two or more matrices",                                                    2, None),
    CatalogEntry("sub",               "-",  "Subtract one matrix from another. Unary `-expr` also negates.",               2,    2),
    CatalogEntry("eq",               "==",  "Check whether two matrices have the same shape and equal entries.",            2,    2),
    CatalogEntry("mult",              "*",  "Multiply two or more matrices (matrix multiplication, or scalar × matrix)",   2, None),
    CatalogEntry("pow",               "^",  "Matrix power — integer exponent (e.g. A^3 = A·A·A)",                          2,    2),
    CatalogEntry("det",              None,  "Determinant of a square matrix",                                              1,    1),
    CatalogEntry("tr",               None,  "Trace of a matrix (sum of diagonal entries)",                                 1,    1),
    CatalogEntry("T",                None,  "Transpose of a matrix",                                                       1,    1),
    CatalogEntry("ref",              None,  "Row echelon form via Gaussian elimination with partial pivoting",              1,    1),
    CatalogEntry("rref",             None,  "Reduced row echelon form",                                                    1,    1),
    CatalogEntry("dist",             None,  "Frobenius distance between two same-shaped matrices or vectors",               2,    2),
    CatalogEntry("angle",            None,  "Angle in radians between two vectors",                                        2,    2),
    CatalogEntry("dot",              None,  "Dot product of two matrices or vectors",                                      2,    2),
    CatalogEntry("qr",               None,  "QR factorization — returns Q (unitary) and R (upper-triangular)",             1,    1),
    CatalogEntry("diag",             None,  "Diagonalization — returns eigenvector matrix P and diagonal matrix D",        1,    1),
    CatalogEntry("solve",            None,  "Solve Ax = b for x given square A and vector/matrix b",                       2,    2),
    CatalogEntry("inv",              None,  "Multiplicative inverse of a square matrix",                                   1,    1),
    CatalogEntry("rank",             None,  "Rank of a matrix",                                                            1,    1),
    CatalogEntry("isIdentity",       None,  "True when the input is an identity matrix",                                   1,    1),
    CatalogEntry("isDiagonal",       None,  "True when all off-diagonal entries are zero",                                 1,    1),
    CatalogEntry("isSymmetric",      None,  "True when the matrix equals its transpose",                                   1,    1),
    CatalogEntry("isUpperTriangular",None,  "True when all entries below the main diagonal are zero",                      1,    1),
    CatalogEntry("isOrthogonal",     None,  "True when A^T · A = I",                                                       1,    1),
    CatalogEntry("isOrthonormal",    None,  "True when the columns of the matrix are orthonormal",                        1,    1),
    CatalogEntry("isIndependent",    None,  "True when the matrix columns are linearly independent",                       1,    1),
    CatalogEntry("lstsq",            None,  "Least-squares solution to a linear matrix equation Ax ≈ b",                   2,    2),
    CatalogEntry("eig",              None,  "Eigenvalues and right eigenvectors of a square matrix",                       1,    1),
    CatalogEntry("schur",            None,  "Schur decomposition — returns unitary Z and quasi-triangular T",              1,    1),
    CatalogEntry("jnf",              None,  "Jordan normal form — returns transform P and Jordan matrix J (uses SymPy)",   1,    1),
    CatalogEntry("norm",             None,  "Frobenius norm of a matrix, or L2 norm of a vector",                         1,    1),
    CatalogEntry("svd",              None,  "Singular Value Decomposition — returns U, S (singular values), and Vt",        1,    1),
    CatalogEntry("gs",               None,  "Gram-Schmidt orthogonalisation (modified) — returns Q whose columns are an orthonormal basis for the column space of A", 1, 1),
    CatalogEntry("isSimilar",        None,  "True when two square matrices are similar (∃ invertible P s.t. B = P⁻¹AP)",                                             2, 2),
    CatalogEntry("lu",               None,  "LU decomposition with partial pivoting — returns permutation P, lower-triangular L, and upper-triangular U such that A = P·L·U", 1, 1),
    CatalogEntry("I",                None,  "Identity matrix of size n×n",                                                                                                         1, 1),
    CatalogEntry("elem_scale",       None,  "Scale elementary matrix — n×n identity with row i multiplied by scalar p (1-based index)",                                              3, 3),
    CatalogEntry("elem_swap",        None,  "Swap elementary matrix — n×n identity with rows i and j exchanged (1-based indices)",                                                 3, 3),
    CatalogEntry("elem_shear",       None,  "Shear elementary matrix — n×n identity with E[i,j] = p; left-multiplying adds p·row j to row i (1-based indices)",                  4, 4),
]

CATALOG_MAP: dict[str, CatalogEntry] = {e.name: e for e in CATALOG}


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _is_scalar(arg) -> bool:
    return isinstance(arg, (int, float, np.integer, np.floating))


def _to_matrix(arg) -> np.ndarray:
    if isinstance(arg, np.ndarray):
        return arg.astype(float)
    if _is_scalar(arg):
        return np.array([[float(arg)]])
    return np.array(arg, dtype=float)


def _result_type(arr: np.ndarray) -> str:
    if arr.ndim == 1 or (arr.ndim == 2 and min(arr.shape) == 1):
        return "vector"
    return "matrix"


def _scalar_result(v) -> dict:
    return {"type": "scalar", "value": float(v)}


def _matrix_result(arr: np.ndarray) -> dict:
    return {"type": _result_type(arr), "value": _serialize_array(arr)}


def _bool_result(v: bool) -> dict:
    return {"type": "boolean", "value": bool(v)}


def _serialize_array(arr: np.ndarray):
    """Convert ndarray to JSON-serializable structure. Complex → {re, im}."""
    if not np.iscomplexobj(arr):
        return arr.tolist()

    def _conv(x):
        re, im = float(x.real), float(x.imag)
        if abs(im) < FLOAT_TOL:
            return round(re, 10)
        return {"re": round(re, 10), "im": round(im, 10)}

    if arr.ndim == 1:
        return [_conv(x) for x in arr]
    return [[_conv(x) for x in row] for row in arr]


# ─── REF / RREF (no NumPy built-in) ──────────────────────────────────────────

def _ref(M: np.ndarray) -> np.ndarray:
    M = M.astype(float).copy()
    rows, cols = M.shape
    pivot_row = 0
    for col in range(cols):
        pivot = next((r for r in range(pivot_row, rows) if abs(M[r, col]) > FLOAT_TOL), None)
        if pivot is None:
            continue
        M[[pivot_row, pivot]] = M[[pivot, pivot_row]]
        M[pivot_row] /= M[pivot_row, col]
        for r in range(pivot_row + 1, rows):
            M[r] -= M[r, col] * M[pivot_row]
        pivot_row += 1
    M[np.abs(M) < FLOAT_TOL] = 0.0
    return M


def _rref(M: np.ndarray) -> np.ndarray:
    M = _ref(M)
    rows, cols = M.shape
    for pivot_row in range(rows - 1, -1, -1):
        pivot_col = next((c for c in range(cols) if abs(M[pivot_row, c] - 1.0) < FLOAT_TOL), None)
        if pivot_col is None:
            continue
        for r in range(pivot_row):
            M[r] -= M[r, pivot_col] * M[pivot_row]
    M[np.abs(M) < FLOAT_TOL] = 0.0
    return M


# ─── Operation implementations ────────────────────────────────────────────────

def _op_add(args):
    first = args[0]
    if _is_scalar(first):
        result = float(first)
        for a in args[1:]:
            if not _is_scalar(a):
                raise ValueError("Cannot add a matrix to a scalar with '+'")
            result += float(a)
        return _scalar_result(result)

    result = _to_matrix(first)
    for a in args[1:]:
        if _is_scalar(a):
            raise ValueError("Cannot add a scalar to a matrix with '+'")
        B = _to_matrix(a)
        if result.shape != B.shape:
            raise ValueError(f"Dimension mismatch for addition: {result.shape} vs {B.shape}")
        result = result + B
    return _matrix_result(result)


def _op_sub(args):
    A, B = args
    if _is_scalar(A) and _is_scalar(B):
        return _scalar_result(float(A) - float(B))
    if _is_scalar(A) or _is_scalar(B):
        raise ValueError("Cannot subtract a scalar and a matrix with '-'")
    A, B = _to_matrix(A), _to_matrix(B)
    if A.shape != B.shape:
        raise ValueError(f"Dimension mismatch for subtraction: {A.shape} vs {B.shape}")
    return _matrix_result(A - B)


def _op_eq(args):
    A, B = args
    if _is_scalar(A) and _is_scalar(B):
        return _bool_result(float(A) == float(B))
    if _is_scalar(A) or _is_scalar(B):
        return _bool_result(False)
    A, B = _to_matrix(A), _to_matrix(B)
    return _bool_result(A.shape == B.shape and bool(np.allclose(A, B, atol=FLOAT_TOL)))


def _op_mult(args):
    # Supports scalar*matrix, matrix*scalar, matrix*matrix chains
    result = args[0]
    for arg in args[1:]:
        if _is_scalar(result) and _is_scalar(arg):
            result = float(result) * float(arg)
        elif _is_scalar(result):
            result = float(result) * _to_matrix(arg)
        elif _is_scalar(arg):
            result = _to_matrix(result) * float(arg)
        else:
            A, B = _to_matrix(result), _to_matrix(arg)
            if A.shape[1] != B.shape[0]:
                raise ValueError(
                    f"Incompatible dimensions for matrix multiplication: {A.shape} × {B.shape}"
                )
            result = np.matmul(A, B)

    if _is_scalar(result):
        return _scalar_result(result)
    return _matrix_result(np.atleast_2d(result) if isinstance(result, np.ndarray) else _to_matrix(result))


def _op_pow(args):
    A, n = args
    if _is_scalar(A) and _is_scalar(n):
        return _scalar_result(float(A) ** float(n))
    if _is_scalar(n):
        A = _to_matrix(A)
        if A.shape[0] != A.shape[1]:
            raise ValueError("Matrix power requires a square matrix")
        n_val = float(n)
        if n_val != int(n_val):
            # Fractional matrix power via scipy
            result = sp_linalg.fractional_matrix_power(A, n_val)
        else:
            result = np.linalg.matrix_power(A, int(n_val))
        return _matrix_result(result)
    raise ValueError("Matrix power exponent must be a scalar")


def _op_det(args):
    A = _to_matrix(args[0])
    if A.shape[0] != A.shape[1]:
        raise ValueError("det requires a square matrix")
    return _scalar_result(np.linalg.det(A))


def _op_tr(args):
    A = _to_matrix(args[0])
    return _scalar_result(np.trace(A))


def _op_T(args):
    A = _to_matrix(args[0])
    return _matrix_result(A.T)


def _op_ref(args):
    A = _to_matrix(args[0])
    return _matrix_result(_ref(A))


def _op_rref(args):
    A = _to_matrix(args[0])
    return _matrix_result(_rref(A))


def _op_dist(args):
    A, B = _to_matrix(args[0]), _to_matrix(args[1])
    if A.shape != B.shape:
        raise ValueError(f"dist requires same-shaped inputs: {A.shape} vs {B.shape}")
    diff = A - B
    return _scalar_result(float(np.sqrt(np.trace(diff.T @ diff))))


def _op_angle(args):
    u = _to_matrix(args[0]).flatten()
    v = _to_matrix(args[1]).flatten()
    if u.shape != v.shape:
        raise ValueError(f"angle requires vectors of equal length: {u.shape} vs {v.shape}")
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    if nu < FLOAT_TOL or nv < FLOAT_TOL:
        raise ValueError("Cannot compute angle with a zero vector")
    cos_t = np.clip(np.dot(u, v) / (nu * nv), -1.0, 1.0)
    return _scalar_result(np.arccos(cos_t))


def _op_dot(args):
    A, B = _to_matrix(args[0]), _to_matrix(args[1])
    # Flatten to 1-D for vector inputs so row/column vectors behave like vectors
    a_flat = A.flatten() if min(A.shape) == 1 else A
    b_flat = B.flatten() if min(B.shape) == 1 else B
    result = np.dot(a_flat, b_flat)
    if np.ndim(result) == 0:
        return _scalar_result(float(result))
    return _matrix_result(np.atleast_2d(result))


def _op_qr(args):
    A = _to_matrix(args[0])
    Q, R = np.linalg.qr(A)
    return {
        "type": "multi_output",
        "outputs": {
            "Q": _matrix_result(Q),
            "R": _matrix_result(R),
        },
    }


def _op_diag(args):
    """Eigendecomposition: A = P D P⁻¹."""
    A = _to_matrix(args[0])
    if A.shape[0] != A.shape[1]:
        raise ValueError("diag (diagonalization) requires a square matrix")
    eigenvalues, P = np.linalg.eig(A)
    if np.linalg.matrix_rank(P) < A.shape[0]:
        raise ValueError("Matrix is defective (not diagonalizable)")
    D = np.diag(eigenvalues)
    return {
        "type": "multi_output",
        "outputs": {
            "P": {"type": "matrix", "value": _serialize_array(P)},
            "D": {"type": "matrix", "value": _serialize_array(D)},
        },
    }


def _op_solve(args):
    A, b = _to_matrix(args[0]), _to_matrix(args[1])
    if A.shape[0] != A.shape[1]:
        raise ValueError("solve requires a square coefficient matrix A")
    if A.shape[0] != b.shape[0]:
        raise ValueError(
            f"Incompatible dimensions: A is {A.shape}, b has {b.shape[0]} rows"
        )
    try:
        x = np.linalg.solve(A, b)
        return _matrix_result(x)
    except np.linalg.LinAlgError as e:
        raise ValueError(f"Cannot solve — matrix may be singular: {e}") from e


def _op_inv(args):
    A = _to_matrix(args[0])
    if A.shape[0] != A.shape[1]:
        raise ValueError("inv requires a square matrix")
    try:
        return _matrix_result(np.linalg.inv(A))
    except np.linalg.LinAlgError:
        raise ValueError("Matrix is singular and cannot be inverted")


def _op_rank(args):
    A = _to_matrix(args[0])
    return {"type": "scalar", "value": int(np.linalg.matrix_rank(A))}


def _op_is_identity(args):
    A = _to_matrix(args[0])
    if A.shape[0] != A.shape[1]:
        return _bool_result(False)
    return _bool_result(np.allclose(A, np.eye(A.shape[0]), atol=FLOAT_TOL))


def _op_is_diagonal(args):
    A = _to_matrix(args[0])
    return _bool_result(np.allclose(A - np.diag(np.diag(A)), 0, atol=FLOAT_TOL))


def _op_is_symmetric(args):
    A = _to_matrix(args[0])
    if A.shape[0] != A.shape[1]:
        return _bool_result(False)
    return _bool_result(np.allclose(A, A.T, atol=FLOAT_TOL))


def _op_is_upper_triangular(args):
    A = _to_matrix(args[0])
    return _bool_result(np.allclose(np.tril(A, -1), 0, atol=FLOAT_TOL))


def _op_is_orthogonal(args):
    A = _to_matrix(args[0])
    if A.shape[0] != A.shape[1]:
        return _bool_result(False)
    return _bool_result(np.allclose(A.T @ A, np.eye(A.shape[0]), atol=FLOAT_TOL))


def _op_is_orthonormal(args):
    A = _to_matrix(args[0])
    cols = A.shape[1]
    return _bool_result(np.allclose(A.T @ A, np.eye(cols), atol=FLOAT_TOL))


def _op_is_independent(args):
    A = _to_matrix(args[0])
    return _bool_result(int(np.linalg.matrix_rank(A)) == A.shape[1])


def _op_lstsq(args):
    A, b = _to_matrix(args[0]), _to_matrix(args[1])
    x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    return _matrix_result(x)


def _op_eig(args):
    A = _to_matrix(args[0])
    if A.shape[0] != A.shape[1]:
        raise ValueError("eig requires a square matrix")
    eigenvalues, eigenvectors = np.linalg.eig(A)
    return {
        "type": "multi_output",
        "outputs": {
            "eigenvalues": {"type": "vector", "value": _serialize_array(eigenvalues)},
            "eigenvectors": {"type": "matrix", "value": _serialize_array(eigenvectors)},
        },
    }


def _op_schur(args):
    A = _to_matrix(args[0])
    if A.shape[0] != A.shape[1]:
        raise ValueError("schur requires a square matrix")
    T, Z = sp_linalg.schur(A)
    return {
        "type": "multi_output",
        "outputs": {
            "Z": {"type": "matrix", "value": _serialize_array(Z)},
            "T": {"type": "matrix", "value": _serialize_array(T)},
        },
    }


def _op_jnf(args):
    A = _to_matrix(args[0])
    if A.shape[0] != A.shape[1]:
        raise ValueError("jnf requires a square matrix")
    try:
        from sympy import Matrix as SMatrix  # type: ignore
        sym_A = SMatrix(A.tolist())
        P_sym, J_sym = sym_A.jordan_form()
        P_np = np.array(P_sym.tolist(), dtype=complex)
        J_np = np.array(J_sym.tolist(), dtype=complex)
        return {
            "type": "multi_output",
            "outputs": {
                "P": {"type": "matrix", "value": _serialize_array(P_np)},
                "J": {"type": "matrix", "value": _serialize_array(J_np)},
            },
        }
    except ImportError:
        raise ValueError("Jordan Normal Form requires SymPy. Install with: pip install sympy")
    except Exception as e:
        raise ValueError(f"Could not compute Jordan Normal Form: {e}") from e


def _op_norm(args):
    A = _to_matrix(args[0])
    return _scalar_result(np.linalg.norm(A))


def _op_svd(args):
    A = _to_matrix(args[0])
    U, s, Vt = np.linalg.svd(A, full_matrices=True)
    # Present S as a 2-D matrix with singular values on the diagonal,
    # same shape as A, so that U @ S @ Vt reconstructs A exactly.
    S = np.zeros_like(A)
    np.fill_diagonal(S, s)
    return {
        "type": "multi_output",
        "outputs": {
            "U":  {"type": "matrix", "value": _serialize_array(U)},
            "S":  {"type": "matrix", "value": _serialize_array(S)},
            "Vt": {"type": "matrix", "value": _serialize_array(Vt)},
        },
    }


def _op_gs(args):
    """Modified Gram-Schmidt orthogonalisation of the columns of A.

    Uses the modified (not classical) algorithm: after each column q_j is
    fixed, its component is subtracted from *all remaining* working vectors,
    not from the original columns.  This halves error accumulation compared
    to the classical formulation.

    Returns Q (m × n), whose columns are mutually orthonormal and span the
    same column space as A.  Raises ValueError if any column is (numerically)
    linearly dependent on the preceding ones.
    """
    A = _to_matrix(args[0])
    m, n = A.shape
    V = A.copy()          # working copy; gets orthogonalised in-place column by column
    Q = np.zeros((m, n))

    for j in range(n):
        # Subtract projections onto already-fixed orthonormal vectors
        for i in range(j):
            V[:, j] -= np.dot(Q[:, i], V[:, j]) * Q[:, i]
        norm = np.linalg.norm(V[:, j])
        if norm < 1e-10:
            raise ValueError(
                f"Column {j + 1} is linearly dependent on the previous column(s); "
                "Gram-Schmidt requires linearly independent input columns"
            )
        Q[:, j] = V[:, j] / norm

    return _matrix_result(Q)


def _op_is_similar(args):
    """Two square matrices A and B are similar iff ∃ invertible P with B = P⁻¹ A P.

    Similarity is determined by comparing Jordan Normal Forms via SymPy.
    Matrices with the same JNF (up to block ordering) are similar.
    """
    A = _to_matrix(args[0])
    B = _to_matrix(args[1])
    if A.shape[0] != A.shape[1] or B.shape[0] != B.shape[1]:
        raise ValueError("isSimilar requires square matrices")
    if A.shape != B.shape:
        return _bool_result(False)
    try:
        from sympy import Matrix as SMatrix, nsimplify  # type: ignore
        SA = SMatrix([[nsimplify(x, rational=True) for x in row] for row in A.tolist()])
        SB = SMatrix([[nsimplify(x, rational=True) for x in row] for row in B.tolist()])
        _, JA = SA.jordan_form()
        _, JB = SB.jordan_form()
        return _bool_result(JA == JB)
    except ImportError:
        raise ValueError("isSimilar requires SymPy. Install with: pip install sympy")
    except Exception as e:
        raise ValueError(f"Could not determine similarity: {e}") from e


def _op_I(args):
    """Return the n×n identity matrix."""
    if not _is_scalar(args[0]):
        raise ValueError("I: n must be a positive integer")
    n = int(round(float(args[0])))
    if n < 1:
        raise ValueError("I: n must be at least 1")
    return _matrix_result(np.eye(n))


def _elem_n_and_row(args, fname):
    """Parse and validate n and one or more 1-based row indices from args."""
    if not _is_scalar(args[0]):
        raise ValueError(f"{fname}: n must be a positive integer")
    n = int(round(float(args[0])))
    if n < 1:
        raise ValueError(f"{fname}: n must be at least 1")

    def _row(idx, name: str) -> int:
        r = int(round(float(idx)))
        if r < 1 or r > n:
            raise ValueError(f"{fname}: row index {name}={r} out of range [1, {n}]")
        return r - 1

    return n, _row


def _op_elem_scale(args):
    """Scale elementary matrix: n×n identity with row i multiplied by p.

    elem_scale(n, p, i) — E[i-1, i-1] = p; all other diagonal entries remain 1.
    Row index i is 1-based.
    """
    n, _row = _elem_n_and_row(args, "elem_scale")
    p = float(args[1])
    i = _row(args[2], "i")
    E = np.eye(n)
    E[i, i] = p
    return _matrix_result(E)


def _op_elem_swap(args):
    """Swap elementary matrix: n×n identity with rows i and j exchanged.

    elem_swap(n, i, j) — left-multiplying A by E swaps rows i and j of A.
    Row indices are 1-based; i and j must be different.
    """
    n, _row = _elem_n_and_row(args, "elem_swap")
    i = _row(args[1], "i")
    j = _row(args[2], "j")
    if i == j:
        raise ValueError("elem_swap: i and j must refer to different rows")
    E = np.eye(n)
    E[[i, j]] = E[[j, i]]
    return _matrix_result(E)


def _op_elem_shear(args):
    """Shear elementary matrix: n×n identity with E[i-1, j-1] = p.

    elem_shear(n, p, i, j) — left-multiplying A by E replaces row i of A
    with row i + p·row j.  Row indices are 1-based; i and j must be different.
    """
    n, _row = _elem_n_and_row(args, "elem_shear")
    p = float(args[1])
    i = _row(args[2], "i")
    j = _row(args[3], "j")
    if i == j:
        raise ValueError("elem_shear: i and j must refer to different rows")
    E = np.eye(n)
    E[i, j] = p
    return _matrix_result(E)


def _op_lu(args):
    """LU decomposition with partial pivoting via SciPy.

    Returns P, L, U such that A = P · L · U, where:
      - P is a permutation matrix (m × m)
      - L is unit lower-triangular (m × k, k = min(m, n))
      - U is upper-triangular (k × n)

    Works for any matrix shape (square or rectangular).
    """
    A = _to_matrix(args[0])
    P, L, U = sp_linalg.lu(A)
    return {
        "type": "multi_output",
        "outputs": {
            "P": _matrix_result(P),
            "L": _matrix_result(L),
            "U": _matrix_result(U),
        },
    }


def _op_neg(args):
    arg = args[0]
    if _is_scalar(arg):
        return _scalar_result(-float(arg))
    A = _to_matrix(arg)
    return _matrix_result(-A)


# ─── Dispatch table ───────────────────────────────────────────────────────────

_OPERATION_FNS: dict = {
    "add":               _op_add,
    "sub":               _op_sub,
    "eq":                _op_eq,
    "mult":              _op_mult,
    "pow":               _op_pow,
    "det":               _op_det,
    "tr":                _op_tr,
    "T":                 _op_T,
    "ref":               _op_ref,
    "rref":              _op_rref,
    "dist":              _op_dist,
    "angle":             _op_angle,
    "dot":               _op_dot,
    "qr":                _op_qr,
    "diag":              _op_diag,
    "solve":             _op_solve,
    "inv":               _op_inv,
    "rank":              _op_rank,
    "isIdentity":        _op_is_identity,
    "isDiagonal":        _op_is_diagonal,
    "isSymmetric":       _op_is_symmetric,
    "isUpperTriangular": _op_is_upper_triangular,
    "isOrthogonal":      _op_is_orthogonal,
    "isOrthonormal":     _op_is_orthonormal,
    "isIndependent":     _op_is_independent,
    "lstsq":             _op_lstsq,
    "eig":               _op_eig,
    "schur":             _op_schur,
    "jnf":               _op_jnf,
    "norm":              _op_norm,
    "svd":               _op_svd,
    "gs":                _op_gs,
    "isSimilar":         _op_is_similar,
    "lu":                _op_lu,
    "I":                 _op_I,
    "elem_scale":        _op_elem_scale,
    "elem_swap":         _op_elem_swap,
    "elem_shear":        _op_elem_shear,
    # internal
    "neg":               _op_neg,
}


def execute(func_name: str, args: list) -> dict:
    """Execute a named operation. Raises ValueError on bad input."""
    entry = CATALOG_MAP.get(func_name)

    # Validate arg count for catalog entries
    if entry is not None:
        n = len(args)
        if n < entry.min_args:
            raise ValueError(
                f"{func_name}() requires at least {entry.min_args} argument(s), got {n}"
            )
        if entry.max_args is not None and n > entry.max_args:
            raise ValueError(
                f"{func_name}() takes at most {entry.max_args} argument(s), got {n}"
            )
    elif func_name != "neg":
        raise ValueError(f"Unknown operation: '{func_name}'")

    fn = _OPERATION_FNS.get(func_name)
    if fn is None:
        raise ValueError(f"No implementation for '{func_name}'")
    return fn(args)
