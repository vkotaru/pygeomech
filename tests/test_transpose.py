from geomech.base.expr import Expression
from geomech.base.vectors import Vector, getVectors
from geomech.base.matrices import Matrix
from geomech.operations.addition import VAdd, MAdd
from geomech.operations.multiplication import VVMul, MVMul
from geomech.operations.transpose import Transpose


class TestTranspose:
    def test_creation(self):
        x = Vector('x')
        t = Transpose(x)
        assert t.type == Expression.VECTOR
        assert str(t) == "(x)'"

    def test_preserves_type(self):
        M = Matrix('M')
        t = Transpose(M)
        assert t.type == Expression.MATRIX

    def test_delta(self):
        x = Vector('x')
        d = Transpose(x).delta()
        assert isinstance(d, Transpose)

    def test_delta_str(self):
        x = Vector('x')
        d = Transpose(x).delta()
        assert str(d) == "(\\delta{x})'"


class TestTransposeAdd:
    def test_vector_add_produces_vadd(self):
        x, y = getVectors(['x', 'y'])
        result = Transpose(x) + Transpose(y)
        assert isinstance(result, VAdd)

    def test_matrix_add(self):
        M = Matrix('M')
        N = Matrix('N')
        result = Transpose(M) + Transpose(N)
        assert isinstance(result, MAdd)


class TestTransposeMul:
    def test_vecT_times_vec(self):
        x, y = getVectors(['x', 'y'])
        result = Transpose(x) * y
        assert isinstance(result, VVMul)
        assert result.type == Expression.SCALAR

    def test_vecT_times_matrix(self):
        x = Vector('x')
        M = Matrix('M')
        result = Transpose(x) * M
        assert isinstance(result, MVMul)
