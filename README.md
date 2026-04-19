# MatPad

> An interactive, browser-based matrix operations workspace powered by a Python numerical backend.

MatPad lets you define matrices, write algebraic expressions using named functions or operator symbols, and instantly compute results — all in a clean, focused interface designed to feel like a math notebook.

---

## Features

- **Compact matrix definition bar** — add, rename, resize, and remove matrices from a persistent chip bar at the top of the workspace
- **Multiline expression editor** — write one operation per line; each line is parsed and evaluated independently
- **Inline autocomplete** — start typing a function name and a ranked dropdown appears; select with keyboard or mouse
- **Dual input modes** — fill a matrix cell-by-cell in a grid, or paste values in NumPy array-literal, semicolon-separated, or plain space-separated format
- **Math expressions as cell values** — enter `2/sqrt(5)`, `pi/4`, `cos(pi/3)` directly in grid cells or text mode; values are evaluated and formatted on entry
- **Rich results panel** — matrices, vectors, scalars, booleans, and multi-output decompositions each rendered in a consistent grid format with dynamic column sizing
- **Copy LaTeX** — one-click copy of all computed results as a LaTeX-formatted string; disabled when there is nothing to copy
- **Save session** — write the current matrices, expressions, and results to a `.tex` file that grows as a running log; subsequent saves append cleanly before `\end{document}`
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
| **Grid** | Click cells and type numbers or math expressions; values are evaluated and formatted when you leave the cell |
| **Text** | Paste or type in any supported format, then click **Apply** |

**Accepted text formats:**

```
[[1, 2, 3], [4, 5, 6]]                        # NumPy array literal
[[2/sqrt(5), 1/sqrt(5)], [-1/sqrt(5), 2/sqrt(5)]]  # expressions in array literal
1 2 3; 4 5 6                                  # semicolon-separated rows
1 2 3                                         # plain paste (newline-separated rows)
4 5 6
```

**Supported math expressions in cell values:**

| Category | Functions / Constants |
|---|---|
| Roots | `sqrt`, `cbrt` |
| Trigonometry | `sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `atan2`, `sinh`, `cosh`, `tanh` |
| Exponential / log | `exp`, `log`, `log2`, `log10` |
| Rounding / misc | `abs`, `ceil`, `floor`, `round`, `sign`, `pow`, `hypot` |
| Constants | `pi`, `e` |
| Operators | `+`, `-`, `*`, `/`, `^` (power) |

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

### 5. Copy results as LaTeX

Click **Copy LaTeX** in the Results panel header to copy all successful results to the clipboard as a ready-to-paste LaTeX string. The button is disabled when there are no results. Each result type is formatted as follows:

| Result type | LaTeX format |
|---|---|
| Scalar | `expr = value` |
| Matrix / vector | `expr = \begin{bmatrix} ... \end{bmatrix}` |
| Boolean | `expr = \text{true}` |
| Multi-output (qr, svd, eig…) | `\begin{aligned}` block with each named output |

### 6. Save session to LaTeX

The **session bar** sits between the expression editor and the workspace panels. Enter a filename (default: `matpad-sessions.tex`) and click **Save Session** to persist the current state to disk.

- If the file does not exist it is created with a full `\documentclass{article}` preamble.
- Each subsequent save appends a new `\section*{MatPad Session — …}` block immediately before `\end{document}`, preserving all prior entries.
- A toast notification confirms the saved path (`sessions/<filename>.tex`).

The generated section contains:
- **Matrices** — each matrix rendered as a `bmatrix`
- **Expressions** — the expression editor content in a `verbatim` block
- **Results** — each successful result as a display-math block (`\[…\]`)

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

### `POST /api/save-session`

```json
{
  "filename": "matpad-sessions.tex",
  "content": "\\section*{MatPad Session --- 2026-04-18 14:32:05}\n..."
}
```

Returns:

```json
{ "path": "sessions/matpad-sessions.tex", "error": null }
```

- The `filename` field is sanitised (non-alphanumeric characters replaced with `_`); a `.tex` extension is appended if missing.
- Files are stored in the `sessions/` directory at the project root.
- A new file receives a complete `\documentclass{article}` document skeleton; subsequent saves insert the new section before `\end{document}`.

---

## Running Tests

```bash
pytest
```

146 tests cover the parser, all operations, and the full API surface including error cases, multi-output results, session save, and edge conditions.
