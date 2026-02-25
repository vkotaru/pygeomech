import pytest
from geomech.base.expr import Expression
from geomech.base.scalars import Scalar, ScalarExpr, Zero, One, Number, getScalars
from geomech.operations.addition import Add
from geomech.operations.multiplication import Mul, SVMul, SMMul
from geomech.operations.geometry import Delta


class TestScalarCreation:
    def test_basic_scalar(self):
        a = Scalar('a')
        assert str(a) == 'a'
        assert a.type == Expression.SCALAR
        assert a.size == (1,)
        assert not a.isConstant
        assert not a.isZero

    def test_constant_scalar(self):
        m = Scalar('m', attr=['Constant'])
        assert m.isConstant
        assert not m.isZero

    def test_zero_scalar(self):
        assert Zero.isZero
        assert Zero.isConstant
        assert str(Zero) == '0'
        assert Zero.value == 0

    def test_one_scalar(self):
        assert One.isOnes
        assert One.isConstant
        assert str(One) == '1'
        assert One.value == 1

    def test_numeric_scalar(self):
        n = Scalar('(3.14)', value=3.14, attr=['Constant'])
        assert n.isNumeric
        assert n.isConstant
        assert n.value == 3.14

    def test_get_scalars_string(self):
        a, b, c = getScalars('a b c')
        assert str(a) == 'a'
        assert str(b) == 'b'
        assert str(c) == 'c'

    def test_get_scalars_list(self):
        a, b = getScalars(['a', 'b'])
        assert str(a) == 'a'
        assert str(b) == 'b'

    def test_get_scalars_with_attr(self):
        m, g = getScalars('m g', attr=['Constant'])
        assert m.isConstant
        assert g.isConstant

    def test_number_float(self):
        n = Number(3.14)
        assert n.isConstant
        assert n.value == 3.14

    def test_number_int(self):
        n = Number(5)
        assert n.isConstant
        assert n.value == 5

    def test_number_string(self):
        n = Number('pi')
        assert n.isConstant
        assert str(n) == 'pi'


class TestScalarArithmetic:
    def test_addition(self):
        a, b = getScalars('a b')
        result = a + b
        assert isinstance(result, Add)
        assert str(result) == '(a+b)'

    def test_triple_addition(self):
        a, b, c = getScalars('a b c')
        result = a + b + c
        assert isinstance(result, Add)

    def test_subtraction(self):
        a, b = getScalars('a b')
        result = a - b
        assert isinstance(result, Add)

    def test_scalar_mul_scalar(self):
        a, b = getScalars('a b')
        result = a * b
        assert isinstance(result, Mul)
        assert result.type == Expression.SCALAR

    def test_scalar_mul_int(self):
        a = Scalar('a')
        result = a * 2
        assert isinstance(result, Mul)

    def test_scalar_mul_vector(self):
        from geomech.base.vectors import Vector
        a = Scalar('a')
        x = Vector('x')
        result = a * x
        assert isinstance(result, SVMul)
        assert result.type == Expression.VECTOR

    def test_scalar_mul_matrix(self):
        from geomech.base.matrices import Matrix
        a = Scalar('a')
        M = Matrix('M')
        result = a * M
        assert isinstance(result, SMMul)
        assert result.type == Expression.MATRIX


class TestScalarOperations:
    def test_delta_variable(self):
        a = Scalar('a')
        d = a.delta()
        assert isinstance(d, Delta)
        assert str(d) == '\\delta{a}'

    def test_delta_constant(self):
        m = Scalar('m', attr=['Constant'])
        d = m.delta()
        assert str(d) == '0'
        assert d.value == 0

    def test_diff_variable(self):
        a = Scalar('a')
        da = a.diff()
        assert str(da) == 'dot_a'

    def test_diff_constant(self):
        m = Scalar('m', attr=['Constant'])
        dm = m.diff()
        assert str(dm) == '0'
        assert dm.isConstant
        assert dm.isZero

    def test_has(self):
        a = Scalar('a')
        b = Scalar('b')
        a2 = Scalar('a')
        assert a.has(a2)
        assert not a.has(b)

    def test_equality(self):
        a1 = Scalar('a')
        a2 = Scalar('a')
        b = Scalar('b')
        assert a1 == a2
        assert not (a1 == b)
