"""Tests for the expression parser and evaluator."""
import math
import pytest
import numpy as np

from backend.parser import tokenize, Parser, evaluate, parse_and_evaluate, ParseError, EvalError
from backend.parser import NumberNode, IdentifierNode, BinaryOpNode, UnaryOpNode, CallNode


# ── Tokenizer ─────────────────────────────────────────────────────────────────

def _tokens(text):
    return [(t.type, t.value) for t in tokenize(text) if t.type not in ('NEWLINE', 'EOF')]

def test_tokenize_simple():
    toks = _tokens("add(A, B)")
    assert toks == [("ID", "add"), ("LPAREN", "("), ("ID", "A"),
                    ("COMMA", ","), ("ID", "B"), ("RPAREN", ")")]

def test_tokenize_operators():
    toks = _tokens("A + B * C ^ 2 == D")
    types = [t for t, _ in toks]
    assert "PLUS" in types
    assert "STAR" in types
    assert "CARET" in types
    assert "EQEQ" in types

def test_tokenize_float():
    toks = _tokens("3.14")
    assert toks == [("NUMBER", 3.14)]

def test_tokenize_scientific():
    toks = _tokens("1e3")
    assert toks == [("NUMBER", 1000.0)]

def test_tokenize_rejects_unknown_char():
    with pytest.raises(ParseError):
        tokenize("A @ B")


# ── Parser: AST structure ─────────────────────────────────────────────────────

def _parse(text):
    tokens = tokenize(text)
    parser = Parser(tokens)
    nodes = parser.parse_program()
    assert len(nodes) == 1
    return nodes[0]

def test_parse_number():
    node = _parse("42")
    assert isinstance(node, NumberNode)
    assert node.value == 42.0

def test_parse_identifier():
    node = _parse("A")
    assert isinstance(node, IdentifierNode)
    assert node.name == "A"

def test_parse_function_call():
    node = _parse("add(A, B)")
    assert isinstance(node, CallNode)
    assert node.name == "add"
    assert len(node.args) == 2

def test_parse_binary_add():
    node = _parse("A + B")
    assert isinstance(node, BinaryOpNode)
    assert node.op == "+"

def test_parse_precedence_mul_over_add():
    # A + B * C  =>  A + (B * C)
    node = _parse("A + B * C")
    assert isinstance(node, BinaryOpNode) and node.op == "+"
    assert isinstance(node.right, BinaryOpNode) and node.right.op == "*"

def test_parse_power_right_associative():
    # A ^ B ^ C  =>  A ^ (B ^ C)
    node = _parse("A ^ B ^ C")
    assert isinstance(node, BinaryOpNode) and node.op == "^"
    assert isinstance(node.right, BinaryOpNode) and node.right.op == "^"

def test_parse_unary_minus_wraps_power():
    # -A ^ 2  =>  -(A ^ 2)  (standard math convention)
    node = _parse("-A ^ 2")
    assert isinstance(node, UnaryOpNode) and node.op == "-"
    assert isinstance(node.operand, BinaryOpNode) and node.operand.op == "^"

def test_parse_parentheses_override_precedence():
    # (A + B) * C
    node = _parse("(A + B) * C")
    assert isinstance(node, BinaryOpNode) and node.op == "*"
    assert isinstance(node.left, BinaryOpNode) and node.left.op == "+"

def test_parse_nested_call():
    node = _parse("mult(A, inv(B))")
    assert isinstance(node, CallNode) and node.name == "mult"
    assert isinstance(node.args[1], CallNode) and node.args[1].name == "inv"

def test_parse_eq_operator():
    node = _parse("A == B")
    assert isinstance(node, BinaryOpNode) and node.op == "=="

def test_parse_rejects_chained_eq():
    with pytest.raises(ParseError):
        _parse("A == B == C")

def test_parse_multiline():
    tokens = tokenize("A + B\nA * B")
    parser = Parser(tokens)
    nodes = parser.parse_program()
    assert len(nodes) == 2

def test_parse_empty_expression():
    tokens = tokenize("")
    parser = Parser(tokens)
    nodes = parser.parse_program()
    assert nodes == []

def test_parse_comment_skipped():
    tokens = tokenize("# just a comment")
    parser = Parser(tokens)
    nodes = parser.parse_program()
    assert nodes == []


# ── Evaluator ─────────────────────────────────────────────────────────────────

M = [[1.0, 2.0], [3.0, 4.0]]
N = [[5.0, 6.0], [7.0, 8.0]]
MATS = {"A": M, "B": N}

def _eval(expr, matrices=None):
    if matrices is None:
        matrices = MATS
    tokens = tokenize(expr)
    parser = Parser(tokens)
    nodes = parser.parse_program()
    return evaluate(nodes[0], matrices)

def test_eval_number():
    assert _eval("5") == 5.0

def test_eval_matrix_ref():
    result = _eval("A")
    np.testing.assert_array_equal(result, np.array(M))

def test_eval_unary_minus_scalar():
    assert _eval("-3") == -3.0

def test_eval_unary_minus_matrix():
    result = _eval("-A")
    expected = -np.array(M)
    np.testing.assert_array_almost_equal(result, expected)

def test_eval_add_matrices():
    result = _eval("A + B")
    expected = np.array(M) + np.array(N)
    np.testing.assert_array_almost_equal(result, expected)

def test_eval_sub_matrices():
    result = _eval("A - B")
    expected = np.array(M) - np.array(N)
    np.testing.assert_array_almost_equal(result, expected)

def test_eval_scalar_mult_left():
    result = _eval("2 * A")
    np.testing.assert_array_almost_equal(result, 2 * np.array(M))

def test_eval_scalar_mult_right():
    result = _eval("A * 2")
    np.testing.assert_array_almost_equal(result, np.array(M) * 2)

def test_eval_matrix_mult():
    result = _eval("A * B")
    np.testing.assert_array_almost_equal(result, np.matmul(np.array(M), np.array(N)))

def test_eval_eq_true():
    result = _eval("A == A")
    assert result is True or result == True

def test_eval_eq_false():
    result = _eval("A == B")
    assert result is False or result == False

def test_eval_parentheses():
    r1 = _eval("(A + B) * A")
    r2 = np.matmul(np.array(M) + np.array(N), np.array(M))
    np.testing.assert_array_almost_equal(r1, r2)

def test_eval_undefined_matrix():
    with pytest.raises(EvalError):
        _eval("C", {})

def test_eval_nested_inv():
    """mult(A, inv(A)) should be approximately the identity."""
    result = _eval("A * inv(A)")
    np.testing.assert_array_almost_equal(result, np.eye(2), decimal=10)

def test_eval_det():
    result = _eval("det(A)")
    expected = np.linalg.det(np.array(M))
    assert abs(result - expected) < 1e-9

def test_eval_transpose():
    result = _eval("T(A)")
    np.testing.assert_array_almost_equal(result, np.array(M).T)

def test_eval_power():
    result = _eval("A ^ 2")
    np.testing.assert_array_almost_equal(result, np.linalg.matrix_power(np.array(M), 2))

def test_eval_multi_output_standalone():
    tokens = tokenize("qr(A)")
    parser = Parser(tokens)
    nodes = parser.parse_program()
    result = evaluate(nodes[0], MATS)
    assert isinstance(result, dict)
    assert result["type"] == "multi_output"
    assert "Q" in result["outputs"]
    assert "R" in result["outputs"]

def test_eval_multi_output_as_subexpr_raises():
    with pytest.raises(EvalError):
        _eval("qr(A) + B")


# ── parse_and_evaluate integration ───────────────────────────────────────────

def test_full_multiline():
    results = parse_and_evaluate("det(A)\ntr(A)", {"A": M})
    assert len(results) == 2
    assert results[0]["error"] is None
    assert results[1]["error"] is None
    assert abs(results[0]["result"]["value"] - np.linalg.det(np.array(M))) < 1e-9
    assert abs(results[1]["result"]["value"] - np.trace(np.array(M))) < 1e-9

def test_full_error_on_invalid_line():
    results = parse_and_evaluate("det(A)\n???\ntr(A)", {"A": M})
    assert results[0]["error"] is None
    assert results[1]["error"] is not None
    assert results[2]["error"] is None

def test_full_comment_skipped():
    results = parse_and_evaluate("# just a comment\ndet(A)", {"A": M})
    assert len(results) == 1

def test_full_undefined_ref():
    results = parse_and_evaluate("det(Z)", {"A": M})
    assert results[0]["error"] is not None
    assert "undefined" in results[0]["error"].lower() or "Z" in results[0]["error"]
