import pytest
from geomech.base.scalars import Scalar, Zero, getScalars
from geomech.base.vectors import Vector, ZeroVector, getVectors
from geomech.base.matrices import Matrix, ZeroMatrix, getMatrices
from geomech.operations.addition import Add, VAdd, MAdd
from geomech.utils.errors import ExpressionMismatchError


class TestNaryFlatten:
    def test_scalar_add_flattens(self):
        a, b, c, d, e = getScalars('a b c d e')
        result = Add(a, b, Add(c, d, e))
        assert isinstance(result, Add)
        assert result.N == 5

    def test_vector_add_flattens(self):
        x, y, z = getVectors(['x', 'y', 'z'])
        result = VAdd(x, VAdd(y, z))
        assert result.N == 3

    def test_matrix_add_flattens(self):
        A, B, C, D = getMatrices('A B C D')
        result = MAdd(A, MAdd(B, C, D))
        assert result.N == 4


class TestNaryDirect:
    def test_nary_scalar_add(self):
        a, b, c = getScalars('a b c')
        result = Add(a, b, c)
        assert result.N == 3
        assert str(result) == '(a+b+c)'

    def test_nary_vector_add(self):
        x, y, z = getVectors(['x', 'y', 'z'])
        result = VAdd(x, y, z)
        assert result.N == 3
        assert str(result) == '(x+y+z)'

    def test_nary_matrix_add(self):
        A, B, C, D = getMatrices('A B C D')
        result = MAdd(A, B, C, D)
        assert result.N == 4
        assert str(result) == '(A+B+C+D)'


class TestAddStr:
    def test_scalar_add_str(self):
        a, b = getScalars('a b')
        result = a + b
        assert str(result) == '(a+b)'

    def test_vector_add_str(self):
        x, y = getVectors(['x', 'y'])
        result = x + y
        assert str(result) == '(x+y)'

    def test_matrix_add_str(self):
        M, N = getMatrices('M N')
        result = M + N
        assert str(result) == '(M+N)'

    def test_scalar_add_flattens_str(self):
        a, b, c, d, e = getScalars('a b c d e')
        result = Add(a, b, Add(c, d, e))
        assert str(result) == '(a+b+c+d+e)'

    def test_vector_add_flattens_str(self):
        x, y, z = getVectors(['x', 'y', 'z'])
        result = VAdd(x, VAdd(y, z))
        assert str(result) == '(x+y+z)'

    def test_matrix_add_flattens_str(self):
        A, B, C, D = getMatrices('A B C D')
        result = MAdd(A, MAdd(B, C, D))
        assert str(result) == '(A+B+C+D)'


class TestAdditionDelta:
    def test_scalar_add_delta(self):
        a, b = getScalars('a b')
        d = (a + b).delta()
        assert isinstance(d, Add)
        assert d.N == 2

    def test_scalar_add_delta_str(self):
        a, b = getScalars('a b')
        d = (a + b).delta()
        assert str(d) == '(\\delta{a}+\\delta{b})'

    def test_vector_add_delta(self):
        x, y = getVectors(['x', 'y'])
        d = (x + y).delta()
        assert isinstance(d, VAdd)
        assert d.N == 2

    def test_vector_add_delta_str(self):
        x, y = getVectors(['x', 'y'])
        d = (x + y).delta()
        assert str(d) == '(\\delta{x}+\\delta{y})'

    def test_matrix_add_delta(self):
        M, N = getMatrices('M N')
        d = (M + N).delta()
        assert isinstance(d, MAdd)
        assert d.N == 2

    def test_matrix_add_delta_str(self):
        M, N = getMatrices('M N')
        d = (M + N).delta()
        assert str(d) == '(\\delta{M}+\\delta{N})'

    def test_all_constant_scalar_delta_is_zero(self):
        m, g = getScalars('m g', attr=['Constant'])
        d = (m + g).delta()
        assert d == Zero

    def test_all_constant_vector_delta_is_zero(self):
        e1 = Vector('e1', attr=['Constant'])
        e2 = Vector('e2', attr=['Constant'])
        d = (e1 + e2).delta()
        assert d == ZeroVector

    def test_all_constant_matrix_delta_is_zero(self):
        A = Matrix('A', attr=['Constant'])
        B = Matrix('B', attr=['Constant'])
        d = (A + B).delta()
        assert d == ZeroMatrix


class TestAdditionDiff:
    def test_scalar_add_diff(self):
        a, b = getScalars('a b')
        d = (a + b).diff()
        assert isinstance(d, Add)
        assert d.N == 2

    def test_scalar_add_diff_str(self):
        a, b = getScalars('a b')
        d = (a + b).diff()
        assert str(d) == '(dot_a+dot_b)'

    def test_vector_add_diff(self):
        x, y = getVectors(['x', 'y'])
        d = (x + y).diff()
        assert isinstance(d, VAdd)
        assert d.N == 2

    def test_vector_add_diff_str(self):
        x, y = getVectors(['x', 'y'])
        d = (x + y).diff()
        assert str(d) == '(dot_x+dot_y)'

    def test_matrix_add_diff(self):
        M, N = getMatrices('M N')
        d = (M + N).diff()
        assert isinstance(d, MAdd)
        assert d.N == 2

    def test_matrix_add_diff_str(self):
        M, N = getMatrices('M N')
        d = (M + N).diff()
        assert str(d) == '(dot_M+dot_N)'

    def test_all_constant_scalar_diff_is_zero(self):
        m, g = getScalars('m g', attr=['Constant'])
        d = (m + g).diff()
        assert d == Zero

    def test_all_constant_vector_diff_is_zero(self):
        e1 = Vector('e1', attr=['Constant'])
        e2 = Vector('e2', attr=['Constant'])
        d = (e1 + e2).diff()
        assert d == ZeroVector

    def test_all_constant_matrix_diff_is_zero(self):
        A = Matrix('A', attr=['Constant'])
        B = Matrix('B', attr=['Constant'])
        d = (A + B).diff()
        assert d == ZeroMatrix


class TestAdditionSizeMismatch:
    """Addition should fail when operand sizes don't match.
    TODO: type mismatch tests raise bare Exception from NaryNode;
    should raise ExpressionMismatchError instead.
    """

    @pytest.mark.xfail(reason="NaryNode does not validate sizes yet")
    def test_vector_add_size_mismatch(self):
        x = Vector('x', size=(3,))
        y = Vector('y', size=(4,))
        with pytest.raises(Exception):
            VAdd(x, y)

    @pytest.mark.xfail(reason="NaryNode does not validate sizes yet")
    def test_matrix_add_size_mismatch(self):
        A = Matrix('A', size=(3, 3))
        B = Matrix('B', size=(4, 4))
        with pytest.raises(Exception):
            MAdd(A, B)

    def test_type_mismatch_scalar_vector(self):
        a = Scalar('a')
        x = Vector('x')
        with pytest.raises(Exception):
            a + x

    def test_type_mismatch_scalar_matrix(self):
        a = Scalar('a')
        M = Matrix('M')
        with pytest.raises(Exception):
            a + M

    def test_type_mismatch_vector_matrix(self):
        x = Vector('x')
        M = Matrix('M')
        with pytest.raises(Exception):
            VAdd(x, M)


class TestSubtraction:
    """Subtraction is implemented as Add with negated term: a - b = a + (-1)*b"""

    def test_scalar_sub(self):
        a, b = getScalars('a b')
        result = a - b
        assert isinstance(result, Add)
        assert result.N == 2

    def test_scalar_sub_str(self):
        a, b = getScalars('a b')
        result = a - b
        assert str(result) == '(a+b(-1))'

    def test_vector_sub(self):
        x, y = getVectors(['x', 'y'])
        result = x - y
        assert isinstance(result, VAdd)
        assert result.N == 2

    def test_vector_sub_str(self):
        x, y = getVectors(['x', 'y'])
        result = x - y
        assert str(result) == '(x+y(-1))'

    def test_matrix_sub(self):
        M, N = getMatrices('M N')
        result = M - N
        assert isinstance(result, MAdd)
        assert result.N == 2

    def test_matrix_sub_str(self):
        M, N = getMatrices('M N')
        result = M - N
        assert str(result) == '(M+N(-1))'

    def test_scalar_sub_type_mismatch(self):
        a = Scalar('a')
        x = Vector('x')
        with pytest.raises(ExpressionMismatchError):
            a - x

    def test_vector_sub_type_mismatch(self):
        x = Vector('x')
        M = Matrix('M')
        with pytest.raises(ExpressionMismatchError):
            x - M

    def test_matrix_sub_type_mismatch(self):
        M = Matrix('M')
        a = Scalar('a')
        with pytest.raises(ExpressionMismatchError):
            M - a
