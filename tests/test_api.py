"""
Integration tests for the FastAPI endpoints.

UI layout (for context, not directly tested here):
  - Matrix definition bar at the top — one chip per matrix (name, rows×cols, remove button)
  - Operation bar — expression textarea + Compute / Clear buttons
  - Panel 1: Matrix Input (grid or text mode)
  - Panel 2: Results
  - Capabilities panel (collapsible, default collapsed)
"""
import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport

from backend.main import app

pytestmark = pytest.mark.asyncio

A = [[1.0, 2.0], [3.0, 4.0]]
B = [[5.0, 6.0], [7.0, 8.0]]
I = [[1.0, 0.0], [0.0, 1.0]]


@pytest_asyncio.fixture
async def client():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


async def post_compute(client, matrices, expression):
    resp = await client.post("/api/compute", json={"matrices": matrices, "expression": expression})
    assert resp.status_code == 200
    return resp.json()


# ── /api/operations ───────────────────────────────────────────────────────────

async def test_operations_endpoint(client):
    resp = await client.get("/api/operations")
    assert resp.status_code == 200
    ops = resp.json()
    assert isinstance(ops, list)
    assert len(ops) > 0
    names = [o["name"] for o in ops]
    for expected in ["add", "det", "inv", "eig", "qr", "schur", "norm"]:
        assert expected in names

async def test_operations_have_required_fields(client):
    resp = await client.get("/api/operations")
    for op in resp.json():
        assert "name" in op
        assert "description" in op
        assert "operator" in op


# ── /api/compute — basic operations ──────────────────────────────────────────

async def test_compute_add(client):
    data = await post_compute(client, {"A": A, "B": B}, "add(A, B)")
    assert data["error"] is None
    r = data["results"][0]
    assert r["error"] is None
    assert r["result"]["type"] == "matrix"
    val = r["result"]["value"]
    assert val[0][0] == pytest.approx(6.0)
    assert val[1][1] == pytest.approx(12.0)

async def test_compute_operator_plus(client):
    data = await post_compute(client, {"A": A, "B": B}, "A + B")
    r = data["results"][0]
    assert r["error"] is None
    assert r["result"]["value"][0][0] == pytest.approx(6.0)

async def test_compute_det(client):
    data = await post_compute(client, {"A": A}, "det(A)")
    r = data["results"][0]
    assert r["error"] is None
    assert r["result"]["type"] == "scalar"
    assert r["result"]["value"] == pytest.approx(-2.0)

async def test_compute_tr(client):
    data = await post_compute(client, {"A": A}, "tr(A)")
    r = data["results"][0]
    assert r["result"]["value"] == pytest.approx(5.0)

async def test_compute_inv(client):
    data = await post_compute(client, {"A": A}, "inv(A)")
    r = data["results"][0]
    assert r["error"] is None
    assert r["result"]["type"] == "matrix"

async def test_compute_transpose(client):
    data = await post_compute(client, {"A": A}, "T(A)")
    r = data["results"][0]
    val = r["result"]["value"]
    assert val[0][1] == pytest.approx(3.0)

async def test_compute_mult_matrix(client):
    data = await post_compute(client, {"A": A, "B": I}, "A * B")
    r = data["results"][0]
    val = r["result"]["value"]
    assert val[0][0] == pytest.approx(1.0)
    assert val[1][1] == pytest.approx(4.0)

async def test_compute_scalar_mult_left(client):
    data = await post_compute(client, {"A": A}, "2 * A")
    r = data["results"][0]
    assert r["error"] is None
    assert r["result"]["value"][0][0] == pytest.approx(2.0)

async def test_compute_scalar_mult_right(client):
    data = await post_compute(client, {"A": A}, "A * 3")
    r = data["results"][0]
    assert r["result"]["value"][0][0] == pytest.approx(3.0)

async def test_compute_eq_true(client):
    data = await post_compute(client, {"A": A}, "A == A")
    r = data["results"][0]
    assert r["result"]["type"] == "boolean"
    assert r["result"]["value"] is True

async def test_compute_eq_false(client):
    data = await post_compute(client, {"A": A, "B": B}, "A == B")
    r = data["results"][0]
    assert r["result"]["value"] is False


# ── Multi-output operations ───────────────────────────────────────────────────

async def test_compute_qr(client):
    data = await post_compute(client, {"A": A}, "qr(A)")
    r = data["results"][0]
    assert r["error"] is None
    assert r["result"]["type"] == "multi_output"
    assert "Q" in r["result"]["outputs"]
    assert "R" in r["result"]["outputs"]

async def test_compute_eig(client):
    data = await post_compute(client, {"A": A}, "eig(A)")
    r = data["results"][0]
    assert r["error"] is None
    assert r["result"]["type"] == "multi_output"
    assert "eigenvalues" in r["result"]["outputs"]
    assert "eigenvectors" in r["result"]["outputs"]

async def test_compute_schur(client):
    data = await post_compute(client, {"A": A}, "schur(A)")
    r = data["results"][0]
    assert r["result"]["type"] == "multi_output"
    assert "Z" in r["result"]["outputs"]
    assert "T" in r["result"]["outputs"]


# ── Multiline expressions ─────────────────────────────────────────────────────

async def test_compute_multiline(client):
    data = await post_compute(client, {"A": A, "B": B}, "det(A)\ntr(B)")
    assert data["error"] is None
    assert len(data["results"]) == 2
    assert data["results"][0]["result"]["type"] == "scalar"
    assert data["results"][1]["result"]["type"] == "scalar"

async def test_compute_multiline_partial_error(client):
    data = await post_compute(client, {"A": A}, "det(A)\n???\ntr(A)")
    assert len(data["results"]) == 3
    assert data["results"][0]["error"] is None
    assert data["results"][1]["error"] is not None
    assert data["results"][2]["error"] is None


# ── Error cases ───────────────────────────────────────────────────────────────

async def test_compute_undefined_matrix(client):
    data = await post_compute(client, {}, "det(A)")
    r = data["results"][0]
    assert r["error"] is not None
    assert "A" in r["error"]

async def test_compute_dim_mismatch(client):
    data = await post_compute(client, {"A": A, "B": [[1.0, 2.0, 3.0]]}, "A + B")
    r = data["results"][0]
    assert r["error"] is not None

async def test_compute_singular_inv(client):
    data = await post_compute(client, {"A": [[1.0, 2.0], [2.0, 4.0]]}, "inv(A)")
    r = data["results"][0]
    assert r["error"] is not None

async def test_compute_empty_expression(client):
    data = await post_compute(client, {"A": A}, "")
    assert data["results"] == []
    assert data["error"] is None

async def test_compute_invalid_matrix_data(client):
    resp = await client.post("/api/compute", json={
        "matrices": {"A": "not a matrix"},
        "expression": "det(A)"
    })
    data = resp.json()
    assert data["error"] is not None

async def test_compute_non_numeric_in_matrix(client):
    resp = await client.post("/api/compute", json={
        "matrices": {"A": [["a", "b"], ["c", "d"]]},
        "expression": "det(A)"
    })
    data = resp.json()
    assert data["error"] is not None


# ── REF / RREF ────────────────────────────────────────────────────────────────

async def test_compute_ref(client):
    data = await post_compute(client, {"A": [[2.0, 1.0, -1.0], [-3.0, -1.0, 2.0], [-2.0, 1.0, 2.0]]}, "ref(A)")
    r = data["results"][0]
    assert r["error"] is None
    assert r["result"]["type"] == "matrix"

async def test_compute_rref(client):
    data = await post_compute(client, {"A": A}, "rref(A)")
    r = data["results"][0]
    assert r["error"] is None
    val = r["result"]["value"]
    # rref of full-rank 2x2 should be identity
    assert val[0][0] == pytest.approx(1.0, abs=1e-9)
    assert val[1][1] == pytest.approx(1.0, abs=1e-9)


# ── Norm / rank / lstsq ───────────────────────────────────────────────────────

async def test_compute_norm(client):
    data = await post_compute(client, {"A": I}, "norm(A)")
    r = data["results"][0]
    assert r["result"]["type"] == "scalar"
    assert r["result"]["value"] == pytest.approx(2.0 ** 0.5, rel=1e-6)

async def test_compute_rank(client):
    data = await post_compute(client, {"A": A}, "rank(A)")
    r = data["results"][0]
    assert r["result"]["value"] == 2

async def test_compute_lstsq(client):
    b = [[5.0], [11.0]]
    data = await post_compute(client, {"A": A, "b": b}, "lstsq(A, b)")
    r = data["results"][0]
    assert r["error"] is None


# ── Boolean predicates via API ────────────────────────────────────────────────

async def test_is_identity_api(client):
    data = await post_compute(client, {"A": I}, "isIdentity(A)")
    assert data["results"][0]["result"]["value"] is True

async def test_is_symmetric_api(client):
    data = await post_compute(client, {"A": [[1.0, 2.0], [2.0, 5.0]]}, "isSymmetric(A)")
    assert data["results"][0]["result"]["value"] is True

async def test_is_upper_triangular_api(client):
    data = await post_compute(client, {"A": [[1.0, 2.0], [0.0, 3.0]]}, "isUpperTriangular(A)")
    assert data["results"][0]["result"]["value"] is True

async def test_is_orthogonal_api(client):
    data = await post_compute(client, {"A": I}, "isOrthogonal(A)")
    assert data["results"][0]["result"]["value"] is True
