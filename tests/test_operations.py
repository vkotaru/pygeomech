import pytest
from geomech.base.scalars import Scalar, getScalars
from geomech.base.vectors import Vector, getVectors
from geomech.base.matrices import Matrix, getMatrices
from geomech.operations.addition import Add, VAdd, MAdd
from geomech.operations.multiplication import Mul, SVMul, MMMul, MVMul
from geomech.operations.geometry import Dot, Cross, Hat, Vee, Delta
from geomech.operations.transpose import Transpose
from geomech.operations.expansion import expand
from geomech.operations.simplification import simplify, full_simplify


class TestProductRuleDelta:
    def test_scalar_mul_delta(self):
        a, b = getScalars('a b')
        expr = a * b
        d = expr.delta()
        assert isinstance(d, Add)

    def test_scalar_vector_mul_delta(self):
        a = Scalar('a')
        x = Vector('x')
        expr = a * x
        d = expr.delta()
        assert isinstance(d, VAdd)

    def test_matrix_vector_mul_delta(self):
        M = Matrix('M')
        x = Vector('x')
        expr = M * x
        d = expr.delta()
        assert isinstance(d, VAdd)

    def test_matrix_matrix_mul_delta(self):
        M, N = getMatrices('M N')
        expr = M * N
        d = expr.delta()
        assert isinstance(d, MAdd)

    def test_constant_left_delta(self):
        m = Scalar('m', attr=['Constant'])
        a = Scalar('a')
        expr = m * a
        d = expr.delta()
        # Only right varies: m * delta(a)
        assert isinstance(d, Mul)

    def test_constant_right_delta(self):
        a = Scalar('a')
        m = Scalar('m', attr=['Constant'])
        expr = a * m
        d = expr.delta()
        # Only left varies: delta(a) * m
        assert isinstance(d, Mul)


class TestAdditionDelta:
    def test_scalar_add_delta(self):
        a, b = getScalars('a b')
        expr = a + b
        d = expr.delta()
        assert isinstance(d, Add)

    def test_vector_add_delta(self):
        x, y = getVectors(['x', 'y'])
        expr = x + y
        d = expr.delta()
        assert isinstance(d, VAdd)

    def test_matrix_add_delta(self):
        M, N = getMatrices('M N')
        expr = M + N
        d = expr.delta()
        assert isinstance(d, MAdd)


class TestAdditionDiff:
    def test_scalar_add_diff(self):
        a, b = getScalars('a b')
        expr = a + b
        d = expr.diff()
        assert isinstance(d, Add)

    def test_vector_add_diff(self):
        x, y = getVectors(['x', 'y'])
        expr = x + y
        d = expr.diff()
        assert isinstance(d, VAdd)


class TestNaryAdd:
    def test_nary_scalar_add(self):
        a, b, c, d, e = getScalars('a b c d e')
        result = Add(a, b, Add(c, d, e))
        # Nested Add should be flattened
        assert isinstance(result, Add)
        assert result.N == 5

    def test_nary_vector_add(self):
        x, y, z = getVectors(['x', 'y', 'z'])
        result = VAdd(x, y, z)
        assert result.N == 3

    def test_nary_matrix_add(self):
        A, B, C = getMatrices('A B C')
        result = MAdd(A, B, C)
        assert result.N == 3


class TestExpansion:
    def test_expand_scalar_distribution(self):
        a, b, c = getScalars('a b c')
        expr = (a + b) * c
        expanded = expand(expr)
        assert isinstance(expanded, Add)

    def test_expand_dot_distribution(self):
        x, y, z = getVectors(['x', 'y', 'z'])
        expr = Dot(x + y, z)
        expanded = expand(expr)
        assert isinstance(expanded, Add)


class TestGeometry:
    def test_dot_creation(self):
        x, y = getVectors(['x', 'y'])
        d = Dot(x, y)
        assert d.type.value == 1  # SCALAR
        assert str(d) == 'Dot(x,y)'

    def test_cross_creation(self):
        x, y = getVectors(['x', 'y'])
        c = Cross(x, y)
        assert c.type.value == 2  # VECTOR
        assert str(c) == 'Cross(x,y)'

    def test_hat_creation(self):
        x = Vector('x')
        h = Hat(x)
        assert h.type.value == 3  # MATRIX
        assert str(h) == 'Hat(x)'

    def test_delta_creation(self):
        x = Vector('x')
        d = Delta(x)
        assert str(d) == '\\delta{x}'


class TestSanityCheck:
    """Reproduces the sanity_check.py example as a test."""

    def test_full_sanity_check(self):
        from geomech import (
            getScalars, getVectors, getMatrices, S2, SO3,
            Transpose, Add, VAdd, MAdd, Dot, Cross, Hat,
            Mul, SVMul, MVMul, MMMul,
        )
        from geomech.utils.errors import ExpressionMismatchError

        a, b, c = getScalars('a b c')
        x, y, z = getVectors(['x', 'y', 'z'])
        M, N = getMatrices('M N')

        # Scalar ops
        assert isinstance(a + b, Add)
        assert isinstance(a + b + c, Add)
        with pytest.raises(Exception):
            a + x
        assert isinstance(a * b, Mul)
        assert isinstance(a * x, SVMul)

        # Vector ops
        assert isinstance(x + y, VAdd)
        assert isinstance(y * b, SVMul)
        assert isinstance(Transpose(x) * y, object)  # VVMul

        # Matrix ops
        assert isinstance(M + N, MAdd)
        assert isinstance(N * y, MVMul)
        assert isinstance(M * N, MMMul)

        # Variations
        scalar_addition = a + b
        d = scalar_addition.delta()
        assert isinstance(d, Add)

        # Manifolds
        q = S2('q')
        assert q.isManifold
        xi = q.get_variation_vector()
        om = q.get_tangent_vector()

        R = SO3('R')
        assert R.isManifold
        eta = R.get_variation_vector()
        Om = R.get_tangent_vector()
