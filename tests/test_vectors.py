import pytest
import numpy as np
from geomech.core.base.types import ExprType
from geomech.core.base.expressions import (
    Vector, S2, TS2, ZeroVector, getVectors, Scalar,
)
from geomech.core.operations.addition import VAdd
from geomech.core.operations.multiplication import SVMul, VVMul
from geomech.core.operations.geometry import Delta, Dot, Cross, Transpose


class TestVectorCreation:
    def test_basic_vector(self):
        x = Vector('x')
        assert str(x) == 'x'
        assert x.type == ExprType.VECTOR
        assert x.size == (3,)
        assert not x.isConstant

    def test_constant_vector(self):
        e3 = Vector('e3', attr=['Constant'])
        assert e3.isConstant

    def test_vector_with_value(self):
        v = Vector('v', value=np.array([0., 0., 1.]))
        assert v.size == (3,)
        assert v.value is not None

    def test_get_vectors_list(self):
        x, y, z = getVectors(['x', 'y', 'z'])
        assert str(x) == 'x'
        assert str(y) == 'y'
        assert str(z) == 'z'

    def test_get_vectors_string(self):
        x, y = getVectors('x y')
        assert str(x) == 'x'
        assert str(y) == 'y'


class TestVectorArithmetic:
    def test_addition(self):
        x, y = getVectors(['x', 'y'])
        result = x + y
        assert isinstance(result, VAdd)
        assert result.type == ExprType.VECTOR

    def test_triple_addition(self):
        x, y, z = getVectors(['x', 'y', 'z'])
        result = x + y + z
        assert isinstance(result, VAdd)

    def test_addition_size_mismatch_fails(self):
        x = Vector('x', size=(3,))
        y = Vector('y', size=(4,))
        with pytest.raises(Exception):
            x + y

    def test_scalar_mul(self):
        x = Vector('x')
        a = Scalar('a')
        result = x * a
        assert isinstance(result, SVMul)
        assert result.type == ExprType.VECTOR

    def test_scalar_left_mul(self):
        x = Vector('x')
        a = Scalar('a')
        result = a * x
        assert isinstance(result, SVMul)
        assert result.type == ExprType.VECTOR

    def test_scalar_mul_commutativity(self):
        x = Vector('x')
        a = Scalar('a')
        left = a * x   # ScalarExpr.__mul__
        right = x * a  # VectorExpr.__mul__
        assert left == right

    def test_int_mul(self):
        x = Vector('x')
        result = x * 2
        assert isinstance(result, SVMul)

    def test_dot_product(self):
        x, y = getVectors(['x', 'y'])
        result = x.dot(y)
        assert isinstance(result, Dot)
        assert result.type == ExprType.SCALAR

    def test_cross_product(self):
        x, y = getVectors(['x', 'y'])
        result = x.cross(y)
        assert isinstance(result, Cross)
        assert result.type == ExprType.VECTOR

    def test_transpose(self):
        x = Vector('x')
        result = x.T()
        assert isinstance(result, Transpose)

    def test_vec_vecT_mul(self):
        x, y = getVectors(['x', 'y'])
        result = x * Transpose(y)
        assert isinstance(result, VVMul)
        assert result.type == ExprType.MATRIX

    def test_vecT_vec_mul(self):
        x, y = getVectors(['x', 'y'])
        result = Transpose(x) * y
        assert isinstance(result, VVMul)
        assert result.type == ExprType.SCALAR


class TestVectorSizeMismatch:
    """Tests that operations fail when vector sizes don't match."""

    def test_dot_size_mismatch_fails(self):
        x = Vector('x', size=(3,))
        y = Vector('y', size=(4,))
        with pytest.raises(Exception):
            Dot(x, y)

    def test_cross_size_mismatch_fails(self):
        x = Vector('x', size=(3,))
        y = Vector('y', size=(4,))
        with pytest.raises(Exception):
            Cross(x, y)

    def test_vec_vecT_mul_different_sizes_allowed(self):
        x = Vector('x', size=(3,))
        y = Vector('y', size=(4,))
        result = x * Transpose(y)
        assert isinstance(result, VVMul)
        assert result.type == ExprType.MATRIX

    def test_vecT_vec_mul_size_mismatch_fails(self):
        x = Vector('x', size=(3,))
        y = Vector('y', size=(4,))
        with pytest.raises(Exception):
            Transpose(x) * y


class TestVectorOperations:
    def test_delta_variable(self):
        x = Vector('x')
        d = x.delta()
        assert isinstance(d, Delta)

    def test_delta_constant(self):
        e3 = Vector('e3', attr=['Constant'])
        d = e3.delta()
        assert str(d) == '0'
        assert d.isConstant

    def test_diff_variable(self):
        x = Vector('x')
        dx = x.t_diff()
        assert str(dx) == 'dot_x'

    def test_diff_constant(self):
        e3 = Vector('e3', attr=['Constant'])
        de3 = e3.t_diff()
        assert de3.isConstant
        assert de3.isZero

    def test_integrate_undoes_diff(self):
        x = Vector('x')
        dx = x.t_diff()
        result = dx.t_integrate()
        assert str(result) == 'x'

    def test_integrate_adds_prefix(self):
        x = Vector('x')
        result = x.t_integrate()
        assert str(result) == 'int_x'


class TestZeroVector:
    def test_zero_vector_is_constant(self):
        assert ZeroVector.isConstant

    def test_zero_vector_is_zero(self):
        assert ZeroVector.isZero


class TestS2Manifold:
    def test_s2_creation(self):
        q = S2('q')
        assert q.isManifold
        assert q.type == ExprType.VECTOR

    def test_s2_variation_vector(self):
        q = S2('q')
        xi = q.get_variation_vector()
        assert isinstance(xi, Vector)
        assert '\\xi' in str(xi)

    def test_s2_tangent_vector(self):
        q = S2('q')
        om = q.get_tangent_vector()
        assert isinstance(om, TS2)
        assert '\\omega' in str(om)

    def test_s2_delta(self):
        q = S2('q')
        dq = q.delta()
        assert isinstance(dq, Cross)

    def test_s2_diff(self):
        q = S2('q')
        dq = q.t_diff()
        assert isinstance(dq, Cross)
