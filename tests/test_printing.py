import pytest
from geomech.core.base.expressions import Scalar, Vector, Matrix, getScalars, getVectors, Zero, ZeroVector
from geomech.core.operations.addition import Add
from geomech.core.operations.multiplication import Mul, SVMul, MVMul
from geomech.core.operations.geometry import Dot, Cross, Hat
from geomech.core.operations.calculus import Variation, TimeDerivative
from geomech.utils.printing import tree_str, print_tree, eom_to_latex


class TestTreeStr:
    def test_leaf_scalar(self):
        a = Scalar('a')
        result = tree_str(a)
        assert "Scalar" in result
        assert "'a'" in result

    def test_leaf_constant(self):
        m = Scalar('m', attr=['Constant'])
        result = tree_str(m)
        assert 'is_constant' in result

    def test_binary_mul(self):
        a, b = getScalars('a b')
        expr = a * b
        result = tree_str(expr)
        assert 'Mul' in result
        assert 'L:' in result
        assert 'R:' in result

    def test_nary_add(self):
        a, b, c = getScalars('a b c')
        expr = a + b + c
        result = tree_str(expr)
        assert 'Add' in result

    def test_unary_hat(self):
        x = Vector('x')
        expr = Hat(x)
        result = tree_str(expr)
        assert 'Hat' in result

    def test_nested_expression(self):
        a = Scalar('a')
        x, y = getVectors(['x', 'y'])
        expr = Mul(a, Dot(x, y))
        result = tree_str(expr)
        assert 'Mul' in result
        assert 'Dot' in result
        assert "'a'" in result


class TestEomToLatex:
    def test_basic_eom_latex(self):
        x = Vector('x')
        var_x = Variation(x)
        a = Scalar('a')
        eqns = {str(var_x): (var_x, a)}
        latex = eom_to_latex(eqns)
        assert r'\documentclass' in latex
        assert r'\begin{equation}' in latex
        assert r'\end{document}' in latex
        assert 'a' in latex
