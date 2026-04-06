from geomech.core.base.expressions import (
    Scalar,
    Vector,
    getScalars,
    getVectors,
)
from geomech.core.operations.calculus import Variation
from geomech.core.operations.geometry import Dot, Hat
from geomech.core.operations.multiplication import Mul
from geomech.utils.printing import eom_to_latex, tree_str


class TestTreeStrTopdown:
    """Tests for the compact top-down tree style (default)."""

    def test_leaf_scalar(self):
        a = Scalar("a")
        result = tree_str(a)
        assert "a" in result

    def test_leaf_constant(self):
        m = Scalar("m", attr=["Constant"])
        result = tree_str(m)
        assert "m*" in result

    def test_binary_mul(self):
        a, b = getScalars("a b")
        expr = a * b
        result = tree_str(expr)
        assert "Mul" in result
        assert "a" in result
        assert "b" in result

    def test_nary_add(self):
        a, b, c = getScalars("a b c")
        expr = a + b + c
        result = tree_str(expr)
        assert "Add" in result

    def test_unary_hat(self):
        x = Vector("x")
        expr = Hat(x)
        result = tree_str(expr)
        assert "Hat" in result

    def test_nested_expression(self):
        a = Scalar("a")
        x, y = getVectors(["x", "y"])
        expr = Mul(a, Dot(x, y))
        result = tree_str(expr)
        assert "Mul" in result
        assert "Dot" in result
        assert "a" in result

    def test_vector_prefix(self):
        x = Vector("x")
        result = tree_str(x)
        assert "v:x" in result

    def test_numeric_value(self):
        s = Scalar("0.5", value=0.5)
        result = tree_str(s)
        assert "0.5" in result

    def test_time_derivative_shortname(self):
        x = Vector("x")
        expr = x.t_diff()
        result = tree_str(expr)
        assert "d/dt" in result


class TestTreeStrIndent:
    """Tests for the verbose indent tree style."""

    def test_leaf_scalar(self):
        a = Scalar("a")
        result = tree_str(a, style="indent")
        assert "Scalar" in result
        assert "'a'" in result

    def test_leaf_constant(self):
        m = Scalar("m", attr=["Constant"])
        result = tree_str(m, style="indent")
        assert "is_constant" in result

    def test_binary_mul(self):
        a, b = getScalars("a b")
        expr = a * b
        result = tree_str(expr, style="indent")
        assert "Mul" in result
        assert "L:" in result
        assert "R:" in result


class TestEomToLatex:
    def test_basic_eom_latex(self):
        x = Vector("x")
        var_x = Variation(x)
        a = Scalar("a")
        eqns = {str(var_x): (var_x, a)}
        latex = eom_to_latex(eqns)
        assert r"\documentclass" in latex
        assert r"\begin{equation}" in latex
        assert r"\end{document}" in latex
        assert "a" in latex
