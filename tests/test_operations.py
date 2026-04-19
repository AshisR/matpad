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
