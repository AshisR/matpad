"""Tests for the NumPy-backed operations implementations."""
import math
import pytest
import numpy as np

from backend.operations import execute, CATALOG_MAP, FLOAT_TOL


# ── Helpers ───────────────────────────────────────────────────────────────────

def mat(data):
    return np.array(data, dtype=float)

def unwrap_matrix(result):
    assert result["type"] in ("matrix", "vector")
    return np.array(result["value"], dtype=float)

def unwrap_scalar(result):
    assert result["type"] == "scalar"
    return result["value"]

def unwrap_bool(result):
    assert result["type"] == "boolean"
    return result["value"]


IDENTITY2 = mat([[1, 0], [0, 1]])
A2        = mat([[1, 2], [3, 4]])
B2        = mat([[5, 6], [7, 8]])
DIAG2     = mat([[2, 0], [0, 3]])
SYM2      = mat([[1, 2], [2, 5]])
UPPER2    = mat([[1, 2], [0, 3]])


# ── add ───────────────────────────────────────────────────────────────────────

def test_add_two():
    r = unwrap_matrix(execute("add", [A2, B2]))
    np.testing.assert_array_almost_equal(r, A2 + B2)

def test_add_three():
    r = unwrap_matrix(execute("add", [A2, B2, IDENTITY2]))
    np.testing.assert_array_almost_equal(r, A2 + B2 + IDENTITY2)

def test_add_dim_mismatch():
    with pytest.raises(ValueError, match="mismatch"):
        execute("add", [A2, mat([[1, 2, 3]])])

def test_add_scalar_scalar():
    r = execute("add", [2.0, 3.0])
    assert r["type"] == "scalar"
    assert r["value"] == 5.0

def test_add_scalar_matrix_raises():
    with pytest.raises(ValueError):
        execute("add", [2.0, A2])


# ── sub ───────────────────────────────────────────────────────────────────────

def test_sub():
    r = unwrap_matrix(execute("sub", [A2, B2]))
    np.testing.assert_array_almost_equal(r, A2 - B2)

def test_sub_dim_mismatch():
    with pytest.raises(ValueError, match="mismatch"):
        execute("sub", [A2, mat([[1]])])

def test_sub_scalar():
    r = execute("sub", [5.0, 3.0])
    assert r["value"] == 2.0


# ── eq ────────────────────────────────────────────────────────────────────────

def test_eq_same():
    assert unwrap_bool(execute("eq", [A2, A2.copy()])) is True

def test_eq_diff():
    assert unwrap_bool(execute("eq", [A2, B2])) is False

def test_eq_scalar():
    assert unwrap_bool(execute("eq", [3.0, 3.0])) is True


# ── mult ──────────────────────────────────────────────────────────────────────

def test_mult_matrix_matrix():
    r = unwrap_matrix(execute("mult", [A2, B2]))
    np.testing.assert_array_almost_equal(r, A2 @ B2)

def test_mult_scalar_left():
    r = unwrap_matrix(execute("mult", [2.0, A2]))
    np.testing.assert_array_almost_equal(r, 2 * A2)

def test_mult_scalar_right():
    r = unwrap_matrix(execute("mult", [A2, 3.0]))
    np.testing.assert_array_almost_equal(r, A2 * 3)

def test_mult_incompatible_dims():
    with pytest.raises(ValueError, match="Incompatible"):
        execute("mult", [mat([[1, 2, 3]]), A2])


# ── pow ───────────────────────────────────────────────────────────────────────

def test_pow_int():
    r = unwrap_matrix(execute("pow", [A2, 2.0]))
    np.testing.assert_array_almost_equal(r, np.linalg.matrix_power(A2, 2))

def test_pow_zero():
    r = unwrap_matrix(execute("pow", [A2, 0.0]))
    np.testing.assert_array_almost_equal(r, np.eye(2))

def test_pow_non_square():
    with pytest.raises(ValueError):
        execute("pow", [mat([[1, 2, 3]]), 2.0])


# ── det ───────────────────────────────────────────────────────────────────────

def test_det_2x2():
    r = unwrap_scalar(execute("det", [A2]))
    assert abs(r - np.linalg.det(A2)) < FLOAT_TOL

def test_det_non_square():
    with pytest.raises(ValueError):
        execute("det", [mat([[1, 2, 3]])])


# ── tr ────────────────────────────────────────────────────────────────────────

def test_tr():
    r = unwrap_scalar(execute("tr", [A2]))
    assert abs(r - np.trace(A2)) < FLOAT_TOL


# ── T (transpose) ─────────────────────────────────────────────────────────────

def test_transpose():
    r = unwrap_matrix(execute("T", [A2]))
    np.testing.assert_array_almost_equal(r, A2.T)


# ── ref / rref ────────────────────────────────────────────────────────────────

def test_ref_upper_triangular():
    A = mat([[2, 1, -1], [-3, -1, 2], [-2, 1, 2]])
    r = unwrap_matrix(execute("ref", [A]))
    # All entries below diagonal should be ~0
    assert np.allclose(np.tril(r, -1), 0, atol=FLOAT_TOL)

def test_rref_identity_block():
    A = mat([[1, 2, 3], [0, 1, 4], [5, 6, 0]])
    r = unwrap_matrix(execute("rref", [A]))
    # Each pivot row should have a 1 at the pivot, 0 elsewhere in that column
    rows, cols = r.shape
    for i in range(rows):
        pivots = [j for j in range(cols) if abs(r[i, j] - 1.0) < FLOAT_TOL]
        if pivots:
            p = pivots[0]
            for row2 in range(rows):
                if row2 != i:
                    assert abs(r[row2, p]) < FLOAT_TOL


# ── dist ──────────────────────────────────────────────────────────────────────

def test_dist_same():
    r = unwrap_scalar(execute("dist", [A2, A2]))
    assert abs(r) < FLOAT_TOL

def test_dist_value():
    r = unwrap_scalar(execute("dist", [A2, B2]))
    diff = A2 - B2
    expected = math.sqrt(np.trace(diff.T @ diff))
    assert abs(r - expected) < FLOAT_TOL

def test_dist_shape_mismatch():
    with pytest.raises(ValueError):
        execute("dist", [A2, mat([[1]])])


# ── angle ─────────────────────────────────────────────────────────────────────

def test_angle_perpendicular():
    u = mat([[1, 0]])
    v = mat([[0, 1]])
    r = unwrap_scalar(execute("angle", [u, v]))
    assert abs(r - math.pi / 2) < FLOAT_TOL

def test_angle_parallel():
    u = mat([[1, 0]])
    r = unwrap_scalar(execute("angle", [u, u]))
    assert abs(r) < FLOAT_TOL


# ── dot ───────────────────────────────────────────────────────────────────────

def test_dot():
    u, v = mat([[1, 2, 3]]), mat([[4, 5, 6]])
    r = execute("dot", [u, v])
    assert abs(unwrap_scalar(r) - np.dot(u.flatten(), v.flatten())) < FLOAT_TOL


# ── qr ────────────────────────────────────────────────────────────────────────

def test_qr_outputs():
    r = execute("qr", [A2])
    assert r["type"] == "multi_output"
    Q = np.array(r["outputs"]["Q"]["value"])
    R = np.array(r["outputs"]["R"]["value"])
    np.testing.assert_array_almost_equal(Q @ R, A2)
    # Q should be orthogonal
    np.testing.assert_array_almost_equal(Q.T @ Q, np.eye(Q.shape[1]), decimal=10)


# ── diag ──────────────────────────────────────────────────────────────────────

def test_diag_diagonal_matrix():
    r = execute("diag", [DIAG2])
    assert r["type"] == "multi_output"
    P = np.array(r["outputs"]["P"]["value"], dtype=complex)
    D = np.array(r["outputs"]["D"]["value"], dtype=complex)
    # A = P D P^-1
    np.testing.assert_array_almost_equal(P @ D @ np.linalg.inv(P), DIAG2, decimal=10)


# ── solve ─────────────────────────────────────────────────────────────────────

def test_solve():
    b = mat([[5], [11]])
    x = unwrap_matrix(execute("solve", [A2, b]))
    np.testing.assert_array_almost_equal(A2 @ x, b)

def test_solve_singular():
    with pytest.raises(ValueError):
        execute("solve", [mat([[1, 2], [2, 4]]), mat([[1], [2]])])


# ── inv ───────────────────────────────────────────────────────────────────────

def test_inv():
    r = unwrap_matrix(execute("inv", [A2]))
    np.testing.assert_array_almost_equal(A2 @ r, np.eye(2), decimal=10)

def test_inv_singular():
    with pytest.raises(ValueError, match="singular"):
        execute("inv", [mat([[1, 2], [2, 4]])])


# ── rank ──────────────────────────────────────────────────────────────────────

def test_rank_full():
    r = execute("rank", [A2])
    assert r["value"] == 2

def test_rank_degenerate():
    r = execute("rank", [mat([[1, 2], [2, 4]])])
    assert r["value"] == 1


# ── Boolean predicates ────────────────────────────────────────────────────────

def test_is_identity_true():
    assert unwrap_bool(execute("isIdentity", [IDENTITY2])) is True

def test_is_identity_false():
    assert unwrap_bool(execute("isIdentity", [A2])) is False

def test_is_diagonal_true():
    assert unwrap_bool(execute("isDiagonal", [DIAG2])) is True

def test_is_diagonal_false():
    assert unwrap_bool(execute("isDiagonal", [A2])) is False

def test_is_symmetric_true():
    assert unwrap_bool(execute("isSymmetric", [SYM2])) is True

def test_is_symmetric_false():
    assert unwrap_bool(execute("isSymmetric", [A2])) is False

def test_is_upper_triangular_true():
    assert unwrap_bool(execute("isUpperTriangular", [UPPER2])) is True

def test_is_upper_triangular_false():
    assert unwrap_bool(execute("isUpperTriangular", [A2])) is False

def test_is_orthogonal_true():
    assert unwrap_bool(execute("isOrthogonal", [IDENTITY2])) is True

def test_is_orthonormal_true():
    assert unwrap_bool(execute("isOrthonormal", [IDENTITY2])) is True

def test_is_independent_full_rank():
    assert unwrap_bool(execute("isIndependent", [A2])) is True

def test_is_independent_rank_deficient():
    assert unwrap_bool(execute("isIndependent", [mat([[1, 2], [2, 4]])])) is False


# ── lstsq ─────────────────────────────────────────────────────────────────────

def test_lstsq_exact():
    b = mat([[5], [11]])
    x = unwrap_matrix(execute("lstsq", [A2, b]))
    np.testing.assert_array_almost_equal(A2 @ x, b, decimal=8)


# ── eig ───────────────────────────────────────────────────────────────────────

def test_eig_outputs():
    r = execute("eig", [DIAG2])
    assert r["type"] == "multi_output"
    vals = r["outputs"]["eigenvalues"]["value"]
    assert len(vals) == 2
    # Eigenvalues of diagonal matrix are the diagonal entries (sorted may differ)
    ev_set = {round(v if isinstance(v, float) else v, 6) for v in vals}
    assert 2.0 in ev_set or abs(min(abs(v - 2.0) if isinstance(v, (int, float)) else 99 for v in vals)) < 1e-6

def test_eig_non_square():
    with pytest.raises(ValueError):
        execute("eig", [mat([[1, 2, 3]])])


# ── schur ─────────────────────────────────────────────────────────────────────

def test_schur_outputs():
    r = execute("schur", [SYM2])
    assert r["type"] == "multi_output"
    Z = np.array(r["outputs"]["Z"]["value"])
    T = np.array(r["outputs"]["T"]["value"])
    np.testing.assert_array_almost_equal(Z @ T @ Z.T, SYM2, decimal=10)


# ── norm ──────────────────────────────────────────────────────────────────────

def test_norm():
    r = unwrap_scalar(execute("norm", [IDENTITY2]))
    assert abs(r - np.linalg.norm(IDENTITY2)) < FLOAT_TOL


# ── svd ───────────────────────────────────────────────────────────────────────

def test_svd_outputs():
    r = execute("svd", [A2])
    assert r["type"] == "multi_output"
    assert {"U", "S", "Vt"} == set(r["outputs"].keys())

def test_svd_reconstruction():
    r = execute("svd", [A2])
    U  = np.array(r["outputs"]["U"]["value"],  dtype=float)
    S  = np.array(r["outputs"]["S"]["value"],  dtype=float)
    Vt = np.array(r["outputs"]["Vt"]["value"], dtype=float)
    np.testing.assert_array_almost_equal(U @ S @ Vt, A2, decimal=10)

def test_svd_singular_values_nonnegative():
    r = execute("svd", [A2])
    S = np.array(r["outputs"]["S"]["value"], dtype=float)
    assert np.all(np.diag(S) >= 0)

def test_svd_rectangular():
    # SVD must work on non-square matrices
    A = mat([[1, 2, 3], [4, 5, 6]])
    r = execute("svd", [A])
    assert r["type"] == "multi_output"
    U  = np.array(r["outputs"]["U"]["value"],  dtype=float)
    S  = np.array(r["outputs"]["S"]["value"],  dtype=float)
    Vt = np.array(r["outputs"]["Vt"]["value"], dtype=float)
    np.testing.assert_array_almost_equal(U @ S @ Vt, A, decimal=10)


# ── neg (internal) ────────────────────────────────────────────────────────────

def test_neg_scalar():
    r = execute("neg", [3.0])
    assert r["value"] == -3.0

def test_neg_matrix():
    r = unwrap_matrix(execute("neg", [A2]))
    np.testing.assert_array_almost_equal(r, -A2)


# ── arg-count validation ──────────────────────────────────────────────────────

def test_det_too_many_args():
    with pytest.raises(ValueError, match="at most"):
        execute("det", [A2, A2])

def test_add_too_few_args():
    with pytest.raises(ValueError, match="at least"):
        execute("add", [A2])

def test_unknown_operation():
    with pytest.raises(ValueError, match="Unknown"):
        execute("foobar", [A2])


# ── Gram-Schmidt (gs) ─────────────────────────────────────────────────────────

def test_gs_columns_orthonormal():
    """Q^T Q must equal I — columns of Q are orthonormal."""
    A = mat([[1, 1, 0], [1, 0, 1], [0, 1, 1]])
    Q = unwrap_matrix(execute("gs", [A]))
    QtQ = Q.T @ Q
    np.testing.assert_array_almost_equal(QtQ, np.eye(3), decimal=10)

def test_gs_same_column_space():
    """Each original column of A must lie in the span of Q."""
    A = mat([[1, 2], [3, 4], [5, 6]])
    Q = unwrap_matrix(execute("gs", [A]))
    # Project each column of A onto Q's column space; residual must be near zero
    for j in range(A.shape[1]):
        proj = Q @ (Q.T @ A[:, j])
        np.testing.assert_array_almost_equal(proj, A[:, j], decimal=10)

def test_gs_identity_input():
    """Gram-Schmidt of the identity matrix returns the identity."""
    I3 = np.eye(3)
    Q = unwrap_matrix(execute("gs", [I3]))
    np.testing.assert_array_almost_equal(np.abs(Q), np.eye(3), decimal=10)

def test_gs_square_result_is_orthogonal():
    """For a square full-rank input, the result Q satisfies Q Q^T = I too."""
    A = mat([[1, 1], [1, 0]])
    Q = unwrap_matrix(execute("gs", [A]))
    np.testing.assert_array_almost_equal(Q @ Q.T, np.eye(2), decimal=10)

def test_gs_rectangular_tall():
    """Tall matrix (m > n): Q is m×n with orthonormal columns."""
    A = mat([[1, 0], [1, 1], [0, 1]])
    Q = unwrap_matrix(execute("gs", [A]))
    assert Q.shape == (3, 2)
    np.testing.assert_array_almost_equal(Q.T @ Q, np.eye(2), decimal=10)

def test_gs_already_orthonormal():
    """Orthonormal input → output matches up to column signs."""
    Q0, _ = np.linalg.qr(np.random.default_rng(42).standard_normal((4, 4)))
    Q = unwrap_matrix(execute("gs", [Q0]))
    # Each column of Q must be ±1 times the corresponding column of Q0
    for j in range(4):
        dot = abs(float(Q[:, j] @ Q0[:, j]))
        assert abs(dot - 1.0) < 1e-9, f"Column {j} not aligned: dot={dot}"

def test_gs_dependent_columns_raises():
    """Linearly dependent columns must raise ValueError."""
    A = mat([[1, 2], [2, 4]])   # col 2 = 2 × col 1
    with pytest.raises(ValueError, match="linearly dependent"):
        execute("gs", [A])

def test_gs_single_column():
    """Single-column input → unit vector."""
    A = mat([[3], [4]])
    Q = unwrap_matrix(execute("gs", [A]))
    np.testing.assert_array_almost_equal(Q, [[0.6], [0.8]], decimal=10)

def test_gs_catalog_entry():
    assert "gs" in CATALOG_MAP
    entry = CATALOG_MAP["gs"]
    assert entry.min_args == 1
    assert entry.max_args == 1


# ── isSimilar ─────────────────────────────────────────────────────────────────

def test_is_similar_same_matrix():
    """A matrix is always similar to itself."""
    assert unwrap_bool(execute("isSimilar", [A2, A2.copy()])) is True

def test_is_similar_true_via_conjugation():
    """B = P⁻¹ A P must be recognised as similar to A."""
    P = mat([[1, 1], [0, 1]])
    B = np.linalg.inv(P) @ A2 @ P
    assert unwrap_bool(execute("isSimilar", [A2, B])) is True

def test_is_similar_identity_only_similar_to_itself():
    """The identity is only similar to itself."""
    assert unwrap_bool(execute("isSimilar", [IDENTITY2, IDENTITY2])) is True
    assert unwrap_bool(execute("isSimilar", [IDENTITY2, A2])) is False

def test_is_similar_diagonalizable_matrices():
    """Two diagonalizable matrices with the same eigenvalues are similar."""
    # DIAG2 has eigenvalues 2 and 3; conjugate it with an invertible P
    P = mat([[2, 1], [1, 1]])
    B = np.linalg.inv(P) @ DIAG2 @ P
    assert unwrap_bool(execute("isSimilar", [DIAG2, B])) is True

def test_is_similar_false_different_eigenvalues():
    """Matrices with different eigenvalues cannot be similar."""
    C = mat([[1, 0], [0, 5]])   # eigenvalues 1, 5
    D = mat([[2, 0], [0, 3]])   # eigenvalues 2, 3
    assert unwrap_bool(execute("isSimilar", [C, D])) is False

def test_is_similar_false_same_eigenvalues_different_jordan():
    """Same eigenvalues but different Jordan structure → not similar."""
    # Both have eigenvalue 2 with algebraic multiplicity 2,
    # but one has a 2×2 Jordan block and the other is diagonal.
    defective  = mat([[2, 1], [0, 2]])   # one 2×2 Jordan block
    diag_equiv = mat([[2, 0], [0, 2]])   # two 1×1 Jordan blocks
    assert unwrap_bool(execute("isSimilar", [defective, diag_equiv])) is False

def test_is_similar_false_different_shape():
    """Matrices of different sizes are never similar."""
    A3 = mat([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    assert unwrap_bool(execute("isSimilar", [A2, A3])) is False

def test_is_similar_non_square_raises():
    """Non-square input must raise ValueError."""
    rect = mat([[1, 2, 3], [4, 5, 6]])
    with pytest.raises(ValueError, match="square"):
        execute("isSimilar", [rect, rect])

def test_is_similar_catalog_entry():
    assert "isSimilar" in CATALOG_MAP
    entry = CATALOG_MAP["isSimilar"]
    assert entry.min_args == 2
    assert entry.max_args == 2


# ── lu ────────────────────────────────────────────────────────────────────────

def test_lu_outputs():
    r = execute("lu", [A2])
    assert r["type"] == "multi_output"
    assert {"P", "L", "U"} == set(r["outputs"].keys())

def test_lu_reconstruction_square():
    """P @ L @ U must reconstruct A exactly."""
    r = execute("lu", [A2])
    P = np.array(r["outputs"]["P"]["value"], dtype=float)
    L = np.array(r["outputs"]["L"]["value"], dtype=float)
    U = np.array(r["outputs"]["U"]["value"], dtype=float)
    np.testing.assert_array_almost_equal(P @ L @ U, A2, decimal=10)

def test_lu_l_unit_lower_triangular():
    """L must be unit lower-triangular (ones on diagonal, zeros above)."""
    r = execute("lu", [A2])
    L = np.array(r["outputs"]["L"]["value"], dtype=float)
    k = min(L.shape)
    assert np.allclose(np.diag(L)[:k], 1.0, atol=1e-10)
    assert np.allclose(np.triu(L, 1), 0, atol=1e-10)

def test_lu_u_upper_triangular():
    """U must be upper-triangular."""
    r = execute("lu", [A2])
    U = np.array(r["outputs"]["U"]["value"], dtype=float)
    assert np.allclose(np.tril(U, -1), 0, atol=1e-10)

def test_lu_p_is_permutation():
    """P must be a permutation matrix: orthogonal with 0/1 entries."""
    r = execute("lu", [A2])
    P = np.array(r["outputs"]["P"]["value"], dtype=float)
    n = P.shape[0]
    np.testing.assert_array_almost_equal(P @ P.T, np.eye(n), decimal=10)
    assert set(np.unique(P)).issubset({0.0, 1.0})

def test_lu_reconstruction_rectangular():
    """LU must work on non-square matrices and reconstruct exactly."""
    A = mat([[1, 2, 3], [4, 5, 6]])   # 2×3
    r = execute("lu", [A])
    P = np.array(r["outputs"]["P"]["value"], dtype=float)
    L = np.array(r["outputs"]["L"]["value"], dtype=float)
    U = np.array(r["outputs"]["U"]["value"], dtype=float)
    np.testing.assert_array_almost_equal(P @ L @ U, A, decimal=10)

def test_lu_shapes_rectangular():
    """For m×n input, P is m×m, L is m×min(m,n), U is min(m,n)×n."""
    A = mat([[1, 2, 3], [4, 5, 6]])   # 2×3
    r = execute("lu", [A])
    P = np.array(r["outputs"]["P"]["value"], dtype=float)
    L = np.array(r["outputs"]["L"]["value"], dtype=float)
    U = np.array(r["outputs"]["U"]["value"], dtype=float)
    assert P.shape == (2, 2)
    assert L.shape == (2, 2)
    assert U.shape == (2, 3)

def test_lu_identity_input():
    """LU of the identity: P = I, L = I, U = I."""
    r = execute("lu", [IDENTITY2])
    P = np.array(r["outputs"]["P"]["value"], dtype=float)
    L = np.array(r["outputs"]["L"]["value"], dtype=float)
    U = np.array(r["outputs"]["U"]["value"], dtype=float)
    np.testing.assert_array_almost_equal(P @ L @ U, IDENTITY2, decimal=10)

def test_lu_singular_matrix():
    """LU decomposition of a singular matrix must still succeed (no error)."""
    A = mat([[1, 2], [2, 4]])   # rank 1
    r = execute("lu", [A])
    P = np.array(r["outputs"]["P"]["value"], dtype=float)
    L = np.array(r["outputs"]["L"]["value"], dtype=float)
    U = np.array(r["outputs"]["U"]["value"], dtype=float)
    np.testing.assert_array_almost_equal(P @ L @ U, A, decimal=10)

def test_lu_catalog_entry():
    assert "lu" in CATALOG_MAP
    entry = CATALOG_MAP["lu"]
    assert entry.min_args == 1
    assert entry.max_args == 1


# ── I (identity) ─────────────────────────────────────────────────────────────

def test_I_is_identity():
    E = unwrap_matrix(execute("I", [3.0]))
    np.testing.assert_array_almost_equal(E, np.eye(3), decimal=10)

def test_I_shape():
    E = unwrap_matrix(execute("I", [5.0]))
    assert E.shape == (5, 5)

def test_I_one_by_one():
    E = unwrap_matrix(execute("I", [1.0]))
    np.testing.assert_array_almost_equal(E, [[1.0]], decimal=10)

def test_I_n_less_than_one():
    with pytest.raises(ValueError, match="at least 1"):
        execute("I", [0.0])

def test_I_catalog_entry():
    assert "I" in CATALOG_MAP
    entry = CATALOG_MAP["I"]
    assert entry.min_args == 1
    assert entry.max_args == 1


# ── elem_scale ────────────────────────────────────────────────────────────────

def test_elem_scale_diagonal():
    """Scale elementary matrix has p on the target diagonal, 1s elsewhere."""
    E = unwrap_matrix(execute("elem_scale", [3.0, 2.5, 2.0]))
    assert abs(E[1, 1] - 2.5) < FLOAT_TOL     # row 2 (1-based) → index 1
    assert abs(E[0, 0] - 1.0) < FLOAT_TOL
    assert abs(E[2, 2] - 1.0) < FLOAT_TOL
    assert np.allclose(np.tril(E, -1), 0, atol=FLOAT_TOL)
    assert np.allclose(np.triu(E,  1), 0, atol=FLOAT_TOL)

def test_elem_scale_integer_p():
    """elem_scale accepts integer p without ambiguity."""
    E = unwrap_matrix(execute("elem_scale", [3.0, 3.0, 1.0]))  # scale row 1 by 3
    assert abs(E[0, 0] - 3.0) < FLOAT_TOL
    assert abs(E[1, 1] - 1.0) < FLOAT_TOL

def test_elem_scale_effect():
    """E_scale @ A must scale the target row by p."""
    A = mat([[1, 2], [3, 4]])
    E = unwrap_matrix(execute("elem_scale", [2.0, 0.5, 1.0]))
    result = E @ A
    np.testing.assert_array_almost_equal(result[0], A[0] * 0.5, decimal=10)
    np.testing.assert_array_almost_equal(result[1], A[1],        decimal=10)

def test_elem_scale_det():
    """det of a scale elementary matrix equals p."""
    E = unwrap_matrix(execute("elem_scale", [3.0, 3.0, 2.0]))
    assert abs(np.linalg.det(E) - 3.0) < 1e-9

def test_elem_scale_row_out_of_range():
    with pytest.raises(ValueError, match="out of range"):
        execute("elem_scale", [3.0, 2.0, 4.0])

def test_elem_scale_n_less_than_one():
    with pytest.raises(ValueError, match="at least 1"):
        execute("elem_scale", [0.0, 2.0, 1.0])

def test_elem_scale_catalog_entry():
    assert "elem_scale" in CATALOG_MAP
    entry = CATALOG_MAP["elem_scale"]
    assert entry.min_args == 3
    assert entry.max_args == 3


# ── elem_swap ─────────────────────────────────────────────────────────────────

def test_elem_swap_structure():
    """Swap elementary matrix is a permutation with the two rows exchanged."""
    E = unwrap_matrix(execute("elem_swap", [3.0, 1.0, 3.0]))
    assert abs(E[0, 2] - 1.0) < FLOAT_TOL   # row 1 now points to col 3
    assert abs(E[2, 0] - 1.0) < FLOAT_TOL   # row 3 now points to col 1
    assert abs(E[1, 1] - 1.0) < FLOAT_TOL   # row 2 unchanged

def test_elem_swap_effect():
    """E_swap @ A must exchange the two target rows of A."""
    A = mat([[1, 2], [3, 4]])
    E = unwrap_matrix(execute("elem_swap", [2.0, 1.0, 2.0]))
    result = E @ A
    np.testing.assert_array_almost_equal(result[0], A[1], decimal=10)
    np.testing.assert_array_almost_equal(result[1], A[0], decimal=10)

def test_elem_swap_det():
    """det of a swap elementary matrix equals -1."""
    E = unwrap_matrix(execute("elem_swap", [3.0, 1.0, 2.0]))
    assert abs(np.linalg.det(E) + 1.0) < 1e-9

def test_elem_swap_same_row():
    with pytest.raises(ValueError, match="different rows"):
        execute("elem_swap", [3.0, 2.0, 2.0])

def test_elem_swap_row_out_of_range():
    with pytest.raises(ValueError, match="out of range"):
        execute("elem_swap", [2.0, 1.0, 3.0])

def test_elem_swap_catalog_entry():
    assert "elem_swap" in CATALOG_MAP
    entry = CATALOG_MAP["elem_swap"]
    assert entry.min_args == 3
    assert entry.max_args == 3


# ── elem_shear ────────────────────────────────────────────────────────────────

def test_elem_shear_structure():
    """Shear elementary matrix has 1s on diagonal and p at (i-1, j-1)."""
    E = unwrap_matrix(execute("elem_shear", [3.0, 5.0, 2.0, 1.0]))
    assert abs(E[1, 0] - 5.0) < FLOAT_TOL
    assert np.allclose(np.diag(E), 1.0, atol=FLOAT_TOL)
    E_copy = E.copy()
    E_copy[1, 0] = 0.0
    assert np.allclose(E_copy, np.eye(3), atol=FLOAT_TOL)

def test_elem_shear_effect():
    """E_shear @ A must add p times row j to row i, leaving others unchanged."""
    A = mat([[1, 2], [3, 4]])
    E = unwrap_matrix(execute("elem_shear", [2.0, 3.0, 1.0, 2.0]))
    result = E @ A
    np.testing.assert_array_almost_equal(result[0], A[0] + 3 * A[1], decimal=10)
    np.testing.assert_array_almost_equal(result[1], A[1],             decimal=10)

def test_elem_shear_det():
    """det of a shear elementary matrix equals 1."""
    E = unwrap_matrix(execute("elem_shear", [3.0, 7.0, 1.0, 2.0]))
    assert abs(np.linalg.det(E) - 1.0) < 1e-9

def test_elem_shear_same_row():
    with pytest.raises(ValueError, match="different rows"):
        execute("elem_shear", [3.0, 5.0, 1.0, 1.0])

def test_elem_shear_row_out_of_range():
    with pytest.raises(ValueError, match="out of range"):
        execute("elem_shear", [3.0, 2.0, 1.0, 4.0])

def test_elem_shear_catalog_entry():
    assert "elem_shear" in CATALOG_MAP
    entry = CATALOG_MAP["elem_shear"]
    assert entry.min_args == 4
    assert entry.max_args == 4
