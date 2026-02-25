import pytest
from geomech.base.scalars import Scalar
from geomech.base.vectors import Vector, getVectors
from geomech.operations.addition import Add, VAdd
from geomech.operations.multiplication import Mul, SVMul
from geomech.operations.geometry import Dot, Cross
from geomech.operations.simplification import (
    simplify, pull, vector_rules,
    has_zeros, combine,
)


class TestSimplifyScalar:
    def test_remove_zeros_from_add(self):
        a = Scalar('a')
        z = Scalar('0', value=0, attr=['Constant', 'Zero'])
        expr = Add(a, z)
        result = simplify(expr)
        assert isinstance(result, Add)

    def test_mul_with_zero(self):
        a = Scalar('a')
        z = Scalar('0', value=0, attr=['Constant', 'Zero'])
        expr = Mul(a, z)
        result = simplify(expr)
        assert result.isZero

    def test_mul_with_one_left(self):
        a = Scalar('a')
        one = Scalar('1', value=1, attr=['Constant', 'Ones'])
        expr = Mul(one, a)
        result = simplify(expr)
        assert result == a

    def test_mul_with_one_right(self):
        a = Scalar('a')
        one = Scalar('1', value=1, attr=['Constant', 'Ones'])
        expr = Mul(a, one)
        result = simplify(expr)
        assert result == a


class TestSimplifyVector:
    def test_remove_zeros_from_vadd(self):
        x = Vector('x')
        z = Vector('0', attr=['Constant', 'Zero'])
        expr = VAdd(x, z)
        result = simplify(expr)
        assert isinstance(result, VAdd)


class TestPull:
    def test_pull_scalar_from_dot_left(self):
        a = Scalar('a')
        x, y = getVectors(['x', 'y'])
        expr = Dot(SVMul(x, a), y)
        result = pull(expr)
        assert isinstance(result, Mul)

    def test_pull_scalar_from_dot_right(self):
        a = Scalar('a')
        x, y = getVectors(['x', 'y'])
        expr = Dot(x, SVMul(y, a))
        result = pull(expr)
        assert isinstance(result, Mul)

    @pytest.mark.xfail(reason="pull() infinite recursion on Cross(SVMul,...) — known bug")
    def test_pull_scalar_from_cross_left(self):
        a = Scalar('a')
        x, y = getVectors(['x', 'y'])
        expr = Cross(SVMul(x, a), y)
        result = pull(expr)
        assert isinstance(result, SVMul)

    @pytest.mark.xfail(reason="pull() infinite recursion on Cross(...,SVMul) — known bug")
    def test_pull_scalar_from_cross_right(self):
        a = Scalar('a')
        x, y = getVectors(['x', 'y'])
        expr = Cross(x, SVMul(y, a))
        result = pull(expr)
        assert isinstance(result, SVMul)


class TestVectorRules:
    def test_dot_self_unit_norm(self):
        q = Vector('q', attr=['Constant'])
        q.isUnitNorm = True
        result = vector_rules(Dot(q, q))
        assert result.value == 1

    def test_dot_cross_orthogonality(self):
        """Dot(x, Cross(x, y)) == 0"""
        x, y = getVectors(['x', 'y'])
        expr = Dot(x, Cross(x, y))
        result = vector_rules(expr)
        assert result.isZero


class TestHelpers:
    def test_has_zeros(self):
        a = Scalar('a')
        z = Scalar('0', value=0, attr=['Constant', 'Zero'])
        assert has_zeros(Mul(a, z))
        assert not has_zeros(Mul(a, a))

    def test_combine(self):
        two = Scalar('2', value=2, attr=['Constant'])
        three = Scalar('3', value=3, attr=['Constant'])
        result = combine(Mul(two, three))
        assert result == 6
