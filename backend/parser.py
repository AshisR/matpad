"""
Expression parser for matrix operations.

Grammar (standard math precedence — ^ binds tighter than unary minus):

    program  := statement* EOF
    statement:= expr NEWLINE | NEWLINE
    expr     := comparison
    comparison := additive ('==' additive)?
    additive := term ( ('+' | '-') term )*
    term     := unary ( '*' unary )*
    unary    := '-' power | power          # unary minus wraps power, so -A^2 = -(A^2)
    power    := primary ( '^' unary )?     # right-associative
    primary  := NUMBER
              | ID '(' arglist ')'
              | ID
              | '(' expr ')'
    arglist  := expr ( ',' expr )*
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .operations import execute as op_execute, CATALOG_MAP

# ─── Token types ──────────────────────────────────────────────────────────────
TK_NUMBER  = "NUMBER"
TK_ID      = "ID"
TK_PLUS    = "PLUS"
TK_MINUS   = "MINUS"
TK_STAR    = "STAR"
TK_CARET   = "CARET"
TK_EQEQ    = "EQEQ"     # ==
TK_LPAREN  = "LPAREN"
TK_RPAREN  = "RPAREN"
TK_COMMA   = "COMMA"
TK_NEWLINE = "NEWLINE"
TK_EOF     = "EOF"

_TOKEN_SPEC = [
    (TK_NUMBER,  r'\d+(?:\.\d*)?(?:[eE][+-]?\d+)?|\.\d+(?:[eE][+-]?\d+)?'),
    (TK_EQEQ,    r'=='),
    (TK_ID,      r'[A-Za-z_][A-Za-z0-9_]*'),
    (TK_PLUS,    r'\+'),
    (TK_MINUS,   r'-'),
    (TK_STAR,    r'\*'),
    (TK_CARET,   r'\^'),
    (TK_LPAREN,  r'\('),
    (TK_RPAREN,  r'\)'),
    (TK_COMMA,   r','),
    (TK_NEWLINE, r'\n'),
    ('SKIP',     r'[ \t\r]+'),
    ('COMMENT',  r'#[^\n]*'),
    ('MISMATCH', r'.'),
]
_MASTER_RE = re.compile(
    '|'.join(f'(?P<{name}>{pat})' for name, pat in _TOKEN_SPEC)
)


# ─── Token ────────────────────────────────────────────────────────────────────

@dataclass
class Token:
    type: str
    value: Any
    line: int
    col: int

    def __repr__(self):
        return f"Token({self.type}, {self.value!r}, L{self.line}:C{self.col})"


# ─── AST nodes ────────────────────────────────────────────────────────────────

@dataclass
class NumberNode:
    value: float

@dataclass
class IdentifierNode:
    name: str

@dataclass
class BinaryOpNode:
    op: str
    left: Any
    right: Any

@dataclass
class UnaryOpNode:
    op: str
    operand: Any

@dataclass
class CallNode:
    name: str
    args: list[Any] = field(default_factory=list)


# ─── Tokenizer ────────────────────────────────────────────────────────────────

class ParseError(Exception):
    def __init__(self, message: str, line: int = 0, col: int = 0):
        super().__init__(message)
        self.line = line
        self.col = col
        self.location = f"line {line}, col {col}" if line else ""

    def __str__(self):
        if self.location:
            return f"{self.location}: {self.args[0]}"
        return self.args[0]


def tokenize(text: str) -> list[Token]:
    tokens: list[Token] = []
    line = 1
    line_start = 0
    for m in _MASTER_RE.finditer(text):
        kind = m.lastgroup
        raw = m.group()
        col = m.start() - line_start + 1
        if kind in ('SKIP', 'COMMENT'):
            continue
        if kind == 'MISMATCH':
            raise ParseError(f"Unexpected character {raw!r}", line, col)
        if kind == TK_NEWLINE:
            tokens.append(Token(TK_NEWLINE, '\n', line, col))
            line += 1
            line_start = m.end()
        elif kind == TK_NUMBER:
            tokens.append(Token(TK_NUMBER, float(raw), line, col))
        else:
            tokens.append(Token(kind, raw, line, col))
    tokens.append(Token(TK_EOF, None, line, 0))
    return tokens


# ─── Recursive-descent parser ─────────────────────────────────────────────────

class Parser:
    def __init__(self, tokens: list[Token]):
        self.tokens = tokens
        self.pos = 0

    @property
    def cur(self) -> Token:
        return self.tokens[self.pos]

    def advance(self) -> Token:
        t = self.tokens[self.pos]
        self.pos += 1
        return t

    def expect(self, typ: str) -> Token:
        if self.cur.type != typ:
            t = self.cur
            raise ParseError(
                f"Expected {typ!r} but found {t.value!r}", t.line, t.col
            )
        return self.advance()

    def _skip_newlines(self):
        while self.cur.type == TK_NEWLINE:
            self.advance()

    # ── Top-level ─────────────────────────────────────────────────────────────

    def parse_program(self) -> list:
        """Return list of (line_source, AST) pairs, one per non-empty line."""
        results = []
        self._skip_newlines()
        while self.cur.type != TK_EOF:
            node = self._parse_expr()
            results.append(node)
            if self.cur.type not in (TK_NEWLINE, TK_EOF):
                t = self.cur
                raise ParseError(f"Unexpected token {t.value!r}", t.line, t.col)
            self._skip_newlines()
        return results

    # ── Expression levels ─────────────────────────────────────────────────────

    def _parse_expr(self):
        return self._parse_comparison()

    def _parse_comparison(self):
        left = self._parse_additive()
        if self.cur.type == TK_EQEQ:
            self.advance()
            right = self._parse_additive()
            # Reject chained equality (A == B == C)
            if self.cur.type == TK_EQEQ:
                t = self.cur
                raise ParseError(
                    "Chained equality is not supported. Use nested eq() for multi-way comparisons.",
                    t.line, t.col,
                )
            return BinaryOpNode("==", left, right)
        return left

    def _parse_additive(self):
        left = self._parse_term()
        while self.cur.type in (TK_PLUS, TK_MINUS):
            op = self.advance().value
            right = self._parse_term()
            left = BinaryOpNode(op, left, right)
        return left

    def _parse_term(self):
        left = self._parse_unary()
        while self.cur.type == TK_STAR:
            self.advance()
            right = self._parse_unary()
            left = BinaryOpNode("*", left, right)
        return left

    def _parse_unary(self):
        # Unary minus wraps power, so -A^2 = -(A^2)  (standard math convention)
        if self.cur.type == TK_MINUS:
            self.advance()
            operand = self._parse_power()
            return UnaryOpNode("-", operand)
        return self._parse_power()

    def _parse_power(self):
        base = self._parse_primary()
        if self.cur.type == TK_CARET:
            self.advance()
            # Right-associative: exponent uses unary (allows -A^-1)
            exp = self._parse_unary()
            return BinaryOpNode("^", base, exp)
        return base

    def _parse_primary(self):
        t = self.cur
        if t.type == TK_NUMBER:
            self.advance()
            return NumberNode(t.value)
        if t.type == TK_LPAREN:
            self.advance()
            self._skip_newlines()
            node = self._parse_expr()
            self._skip_newlines()
            self.expect(TK_RPAREN)
            return node
        if t.type == TK_ID:
            self.advance()
            if self.cur.type == TK_LPAREN:
                # Function call
                self.advance()
                args = []
                self._skip_newlines()
                if self.cur.type != TK_RPAREN:
                    args.append(self._parse_expr())
                    while self.cur.type == TK_COMMA:
                        self.advance()
                        self._skip_newlines()
                        args.append(self._parse_expr())
                self._skip_newlines()
                self.expect(TK_RPAREN)
                return CallNode(t.value, args)
            # Plain identifier — matrix/variable reference
            return IdentifierNode(t.value)
        raise ParseError(f"Unexpected token {t.value!r}", t.line, t.col)


# ─── Evaluator ────────────────────────────────────────────────────────────────

class EvalError(Exception):
    pass


_OP_TO_FN = {"+": "add", "-": "sub", "*": "mult", "^": "pow", "==": "eq"}


def _unwrap_single(result: dict):
    """
    Convert a single-output operation result dict into a Python-native value
    (float, bool, or ndarray).  Must not be called on multi_output results.
    """
    if result["type"] == "scalar":
        return result["value"]
    if result["type"] == "boolean":
        return result["value"]
    if result["type"] in ("matrix", "vector"):
        v = result["value"]
        return np.array(v, dtype=float) if not _has_complex(v) else np.array(_to_complex(v), dtype=complex)
    return result["value"]


def _unwrap(result: dict):
    """
    Unwrap a result dict for use as an operand inside a larger expression.
    Raises EvalError if the result is multi-output (cannot be used inline).
    """
    if result["type"] == "multi_output":
        names = list(result["outputs"].keys())
        raise EvalError(
            f"This operation returns multiple outputs ({', '.join(names)}) "
            "and cannot be used as a sub-expression. "
            "Use it as a standalone statement."
        )
    return _unwrap_single(result)


def _has_complex(v) -> bool:
    if isinstance(v, dict):
        return True
    if isinstance(v, list):
        return any(_has_complex(x) for x in v)
    return False


def _to_complex(v):
    if isinstance(v, dict):
        return complex(v["re"], v["im"])
    if isinstance(v, list):
        return [_to_complex(x) for x in v]
    return v


def evaluate(node, matrices: dict):
    """Evaluate an AST node, returning a Python scalar, ndarray, bool, or result dict."""
    if isinstance(node, NumberNode):
        return node.value

    if isinstance(node, IdentifierNode):
        if node.name in matrices:
            data = matrices[node.name]
            return np.array(data, dtype=float)
        raise EvalError(f"Undefined matrix or variable: '{node.name}'")

    if isinstance(node, UnaryOpNode):
        if node.op == "-":
            val = evaluate(node.operand, matrices)
            if isinstance(val, dict):
                val = _unwrap(val)   # raises if multi-output
            result = op_execute("neg", [val])
            return _unwrap_single(result)
        raise EvalError(f"Unknown unary operator: '{node.op}'")

    if isinstance(node, BinaryOpNode):
        left = evaluate(node.left, matrices)
        right = evaluate(node.right, matrices)
        # Unwrap any single-output op results so they become Python-native values.
        # _unwrap raises EvalError if either side is a multi-output result.
        if isinstance(left, dict):
            left = _unwrap(left)
        if isinstance(right, dict):
            right = _unwrap(right)
        fn_name = _OP_TO_FN.get(node.op)
        if fn_name is None:
            raise EvalError(f"Unknown operator: '{node.op}'")
        result = op_execute(fn_name, [left, right])
        return _unwrap_single(result)

    if isinstance(node, CallNode):
        args = []
        for i, arg_node in enumerate(node.args):
            val = evaluate(arg_node, matrices)
            # Unwrap single-output dicts; reject multi-output
            if isinstance(val, dict):
                val = _unwrap(val)   # raises if multi-output
            args.append(val)
        result = op_execute(node.name, args)
        # Multi-output stays as a dict (handled by _to_serializable_result at top level).
        # Single-output is unwrapped so it can be used as an operand in a parent expression.
        if result.get("type") == "multi_output":
            return result
        return _unwrap_single(result)

    raise EvalError(f"Unknown AST node type: {type(node).__name__}")


def _to_serializable_result(value) -> dict:
    """Convert a raw evaluated value to a typed result dict."""
    # Already a typed result dict (from op_execute)
    if isinstance(value, dict) and "type" in value:
        return value
    if isinstance(value, bool):
        return {"type": "boolean", "value": value}
    if isinstance(value, (int, float, np.integer, np.floating)):
        return {"type": "scalar", "value": float(value)}
    if isinstance(value, np.ndarray):
        from .operations import _serialize_array, _result_type
        return {"type": _result_type(value), "value": _serialize_array(value)}
    return {"type": "unknown", "value": str(value)}


# ─── Public API ───────────────────────────────────────────────────────────────

def parse_and_evaluate(expression: str, matrices: dict) -> list[dict]:
    """
    Parse and evaluate a potentially multiline expression string.

    Each non-empty, non-comment line is evaluated independently against the
    provided matrices dict ({name: [[row…], …]}).

    Returns a list of result records:
        [{"line": int, "expr": str, "result": dict|None, "error": str|None}, …]
    """
    lines = expression.split('\n')
    results = []

    for line_num, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line or line.startswith('#'):
            continue

        try:
            tokens = tokenize(line)
            parser = Parser(tokens)
            nodes = parser.parse_program()

            if not nodes:
                continue

            if len(nodes) > 1:
                raise ParseError("Expected one expression per line")

            raw = evaluate(nodes[0], matrices)
            result_dict = _to_serializable_result(raw)

            results.append({
                "line": line_num,
                "expr": line,
                "result": result_dict,
                "error": None,
            })

        except (ParseError, EvalError, ValueError) as exc:
            results.append({
                "line": line_num,
                "expr": line,
                "result": None,
                "error": str(exc),
            })
        except Exception as exc:
            results.append({
                "line": line_num,
                "expr": line,
                "result": None,
                "error": f"Unexpected computation error: {exc}",
            })

    return results
