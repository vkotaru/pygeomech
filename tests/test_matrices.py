import pytest
from geomech.base.expr import Expression
from geomech.base.matrices import (
    Matrix, MatrixExpr, SO3, ZeroMatrix, IdentityMatrix, O, I, getMatrices,
)
from geomech.base.scalars import Scalar
from geomech.base.vectors import Vector, TSO3
from geomech.operations.addition import MAdd
from geomech.operations.multiplication import SMMul, MVMul, MMMul
from geomech.operations.geometry import Delta


class TestMatrixCreation:
    def test_basic_matrix(self):
        M = Matrix('M')
        assert str(M) == 'M'
        assert M.type == Expression.MATRIX
        assert M.size == (3, 3)
        assert not M.isConstant

    def test_constant_matrix(self):
        J = Matrix('J', attr=['Constant'])
        assert J.isConstant

    def test_symmetric_matrix(self):
        J = Matrix('J', attr=['Constant', 'SymmetricMatrix'])
        assert J.isSymmetric
        assert J.isConstant

    def test_zero_matrix(self):
        assert ZeroMatrix.isZero
        assert ZeroMatrix.isConstant
        assert str(ZeroMatrix) == '0'

    def test_identity_matrix(self):
        assert IdentityMatrix.isConstant
        assert str(IdentityMatrix) == 'I'

    def test_aliases(self):
        assert O is ZeroMatrix
        assert I is IdentityMatrix

    def test_get_matrices(self):
        M, N = getMatrices('M N')
        assert str(M) == 'M'
        assert str(N) == 'N'

    def test_get_matrices_list(self):
        A, B = getMatrices(['A', 'B'])
        assert str(A) == 'A'
        assert str(B) == 'B'


class TestMatrixArithmetic:
    def test_addition(self):
        M, N = getMatrices('M N')
        result = M + N
        assert isinstance(result, MAdd)
        assert result.type == Expression.MATRIX

    @pytest.mark.xfail(reason="NaryNode does not validate sizes yet")
    def test_addition_size_mismatch_fails(self):
        A = Matrix('A', size=(3, 3))
        B = Matrix('B', size=(4, 4))
        with pytest.raises(Exception):
            A + B

    def test_scalar_mul(self):
        M = Matrix('M')
        a = Scalar('a')
        result = M * a
        assert isinstance(result, SMMul)

    def test_scalar_left_mul(self):
        M = Matrix('M')
        a = Scalar('a')
        result = a * M
        assert isinstance(result, SMMul)

    def test_scalar_mul_commutativity(self):
        M = Matrix('M')
        a = Scalar('a')
        left = a * M   # ScalarExpr.__mul__
        right = M * a  # MatrixExpr.__mul__
        assert left == right

    def test_int_mul(self):
        M = Matrix('M')
        result = M * 2
        assert isinstance(result, SMMul)

    def test_vector_mul(self):
        M = Matrix('M')
        x = Vector('x')
        result = M * x
        assert isinstance(result, MVMul)
        assert result.type == Expression.VECTOR

    def test_matrix_mul(self):
        M, N = getMatrices('M N')
        result = M * N
        assert isinstance(result, MMMul)
        assert result.type == Expression.MATRIX


class TestMatrixOperations:
    def test_delta_variable(self):
        M = Matrix('M')
        d = M.delta()
        assert isinstance(d, Delta)

    def test_delta_constant(self):
        J = Matrix('J', attr=['Constant'])
        d = J.delta()
        assert str(d) == 'O'
        assert d.isConstant

    def test_diff_variable(self):
        M = Matrix('M')
        dM = M.diff()
        assert str(dM) == 'dot_M'

    def test_diff_constant(self):
        J = Matrix('J', attr=['Constant'])
        dJ = J.diff()
        assert dJ.isConstant
        assert dJ.isZero


class TestSO3Manifold:
    def test_so3_creation(self):
        R = SO3('R')
        assert R.isManifold
        assert R.type == Expression.MATRIX

    def test_so3_variation_vector(self):
        R = SO3('R')
        eta = R.get_variation_vector()
        assert isinstance(eta, Vector)
        assert '\\eta' in str(eta)

    def test_so3_tangent_vector(self):
        R = SO3('R')
        Om = R.get_tangent_vector()
        assert isinstance(Om, TSO3)
        assert '\\Omega' in str(Om)

    def test_so3_delta(self):
        R = SO3('R')
        dR = R.delta()
        assert isinstance(dR, MMMul)

    def test_so3_diff(self):
        R = SO3('R')
        dR = R.diff()
        assert isinstance(dR, MMMul)
