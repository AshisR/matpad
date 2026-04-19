# MatPad

> An interactive, browser-based matrix operations workspace powered by a Python numerical backend.

MatPad lets you define matrices, write algebraic expressions using named functions or operator symbols, and instantly compute results — all in a clean, focused interface designed to feel like a math notebook.

---

## Features

- **Compact matrix definition bar** — add, rename, resize, and remove matrices from a persistent chip bar at the top of the workspace
- **Multiline expression editor** — write one operation per line; each line is parsed and evaluated independently
- **Inline autocomplete** — start typing a function name and a ranked dropdown appears; select with keyboard or mouse
- **Dual input modes** — fill a matrix cell-by-cell in a grid, or paste values in NumPy array-literal, semicolon-separated, or plain space-separated format
- **Rich results panel** — matrices, vectors, scalars, booleans, and multi-output decompositions each rendered in a consistent grid format
- **Collapsible panels** — Matrix Input, Results, and Capabilities panels collapse independently; state persists across page refreshes
- **Capabilities reference** — searchable table of all supported operations with fuzzy matching on name, operator, and description

---

## Getting Started

### Requirements

- Python 3.7+
- pip

### Install & Run

```bash
# From the project root
pip install -r backend/requirements.txt
python run.py
```

Then open [http://localhost:8000](http://localhost:8000) in your browser.

---

## Using MatPad

### 1. Define your matrices

The **matrix definition bar** at the top of the workspace shows one chip per matrix. Each chip has:

- An **editable name** — click it to rename (letters, digits, and underscores; must start with a letter)
- **Row × column inputs** — change dimensions inline; existing values are preserved where they fit
- A **× remove button** — disabled when only one matrix remains

Click **+ Add Matrix** to append a new matrix with the next available letter and 2 × 2 dimensions.

### 2. Enter values

Select a matrix chip, then fill in values using the **Matrix Input** panel:

| Mode | How to enter data |
|---|---|
| **Grid** | Click cells and type numbers; navigate with arrow keys, Tab, or Enter |
| **Text** | Paste or type in NumPy array-literal, semicolon-separated rows, or plain space-separated rows, then click **Apply** |

**Accepted text formats:**

```
[[1, 2, 3], [4, 5, 6]]       # NumPy array literal
1 2 3; 4 5 6                  # semicolon-separated rows
1 2 3                         # plain paste (newline-separated rows)
4 5 6
```

### 3. Write expressions

Type one or more operations into the **expression editor**, one per line:

```
det(A)
inv(A) * B
A + B
eig(A)
isSymmetric(A)
```

Start typing a function name to trigger **autocomplete** — use arrow keys to navigate, Enter or Tab to insert.

### 4. Compute

Click **Compute** (or press Enter with an active item selected in the autocomplete). Each line is evaluated and the result appears in the **Results** panel alongside the expression that produced it.

---

## Expression Syntax

MatPad supports named function calls and overloaded operator symbols interchangeably.

### Operator precedence (high → low)

| Level | Operators | Example |
|---|---|---|
| 1 — highest | `^` (power) | `A^2` |
| 2 | unary `-` (negation) | `-A` |
| 3 | `*` (multiplication) | `A * B`, `2 * A`, `A * 3` |
| 4 | `+`, `-` (addition / subtraction) | `A + B - C` |
| 5 — lowest | `==` (equality check) | `A == B` |

Parentheses override precedence in the usual way: `(A + B) * C`.

### Nesting and composition

Functions and operators compose freely:

```
det(inv(A))
tr(A + B)
A * inv(B) + C
norm(A - B)
```

---

## Supported Operations

### Arithmetic

| Function | Operator | Description |
|---|---|---|
| `add` | `+` | Add two or more matrices |
| `sub` | `-` | Subtract matrices; unary `-` negates an expression |
| `mult` | `*` | Multiply matrices or scale by a scalar (`2*A` or `A*2`) |
| `pow` | `^` | Raise a matrix to an integer power |
| `eq` | `==` | Test shape and entry equality; returns a boolean |

### Core matrix operations

| Function | Description |
|---|---|
| `det(A)` | Determinant |
| `tr(A)` | Trace |
| `T(A)` | Transpose |
| `inv(A)` | Multiplicative inverse |
| `rank(A)` | Rank |
| `norm(A)` | Matrix or vector norm |
| `ref(A)` | Row echelon form |
| `rref(A)` | Reduced row echelon form |

### Factorizations & decompositions

| Function | Outputs | Description |
|---|---|---|
| `qr(A)` | `Q`, `R` | QR factorization |
| `eig(A)` | `eigenvalues`, `eigenvectors` | Eigendecomposition |
| `svd(A)` | `U`, `S`, `Vt` | Singular Value Decomposition |
| `schur(A)` | `Z`, `T` | Schur decomposition |
| `jnf(A)` | `P`, `J` | Jordan normal form |
| `diag(A)` | `P`, `D` | Diagonalization |

### Solving & estimation

| Function | Description |
|---|---|
| `solve(A, b)` | Exact solution to `Ax = b` |
| `lstsq(A, b)` | Least-squares solution |

### Geometry

| Function | Description |
|---|---|
| `dot(A, B)` | Dot product |
| `dist(A, B)` | Frobenius distance between matrices or vectors |
| `angle(A, B)` | Angle in radians between two vectors |

### Boolean predicates

| Function | Returns `true` when… |
|---|---|
| `isIdentity(A)` | A is an identity matrix |
| `isDiagonal(A)` | All off-diagonal entries are zero |
| `isSymmetric(A)` | A equals its transpose |
| `isUpperTriangular(A)` | All entries below the main diagonal are zero |
| `isOrthogonal(A)` | Aᵀ A = I |
| `isOrthonormal(A)` | Columns are orthonormal |
| `isIndependent(A)` | Columns are linearly independent |

---

## Project Structure

```
matpad/
├── backend/
│   ├── main.py          # FastAPI app — REST endpoints and static file serving
│   ├── operations.py    # Operation catalog and NumPy/SciPy/SymPy implementations
│   ├── parser.py        # Tokenizer, recursive-descent parser, and evaluator
│   └── requirements.txt
├── frontend/
│   ├── index.html
│   ├── css/style.css
│   └── js/app.js
├── tests/
│   ├── test_api.py
│   ├── test_operations.py
│   └── test_parser.py
├── spec/
│   ├── spec.md          # Feature specification
│   └── Operations.md    # Operations reference
├── pytest.ini
└── run.py
```

---

## API

The frontend communicates with two endpoints:

### `GET /api/operations`

Returns the full operation catalog used to populate the Capabilities panel and autocomplete.

### `POST /api/compute`

```json
{
  "matrices": {
    "A": [[1, 2], [3, 4]],
    "B": [[5, 6], [7, 8]]
  },
  "expression": "det(A)\nA * inv(A)"
}
```

Returns one result object per non-empty expression line:

```json
{
  "error": null,
  "results": [
    { "line": 1, "expr": "det(A)", "error": null, "result": { "type": "scalar", "value": -2.0 } },
    { "line": 2, "expr": "A * inv(A)", "error": null, "result": { "type": "matrix", "value": [[1,0],[0,1]] } }
  ]
}
```

---

## Running Tests

```bash
pytest
```

135 tests cover the parser, all operations, and the full API surface including error cases, multi-output results, and edge conditions.
