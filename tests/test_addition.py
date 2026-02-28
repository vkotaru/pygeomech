import pytest
from geomech.core.base.expressions import (
    Scalar, Vector, Matrix, Zero, ZeroVector, ZeroMatrix,
    getScalars, getVectors, getMatrices,
)
from geomech.core.operations.addition import Add, VAdd, MAdd
from geomech.core.base.types import ExprType
from geomech.utils.errors import ExpressionMismatchError, SizeMismatchError


# ---------------------------------------------------------------------------
# Type
# ---------------------------------------------------------------------------

class TestAddType:
    def test_add_type(self):
        a, b = getScalars('a b')
        assert Add(a, b).type == ExprType.SCALAR

    def test_vadd_type(self):
        x, y = getVectors(['x', 'y'])
        assert VAdd(x, y).type == ExprType.VECTOR

    def test_madd_type(self):
        A, B = getMatrices('A B')
        assert MAdd(A, B).type == ExprType.MATRIX


# ---------------------------------------------------------------------------
# Construction & arity & nodes
# ---------------------------------------------------------------------------

class TestBinaryAdd:
    def test_scalar_binary(self):
        a, b = getScalars('a b')
        result = Add(a, b)
        assert isinstance(result, Add)
        assert result.arity == 2
        assert result.nodes[0] == a
        assert result.nodes[1] == b

    def test_vector_binary(self):
        x, y = getVectors(['x', 'y'])
        result = VAdd(x, y)
        assert isinstance(result, VAdd)
        assert result.arity == 2
        assert result.nodes[0] == x
        assert result.nodes[1] == y

    def test_matrix_binary(self):
        A, B = getMatrices('A B')
        result = MAdd(A, B)
        assert isinstance(result, MAdd)
        assert result.arity == 2
        assert result.nodes[0] == A
        assert result.nodes[1] == B


class TestNaryDirect:
    def test_scalar_3(self):
        a, b, c = getScalars('a b c')
        result = Add(a, b, c)
        assert result.arity == 3
        assert str(result) == '(a+b+c)'

    def test_vector_3(self):
        x, y, z = getVectors(['x', 'y', 'z'])
        result = VAdd(x, y, z)
        assert result.arity == 3
        assert str(result) == '(x+y+z)'

    def test_matrix_4(self):
        A, B, C, D = getMatrices('A B C D')
        result = MAdd(A, B, C, D)
        assert result.arity == 4
        assert str(result) == '(A+B+C+D)'


# ---------------------------------------------------------------------------
# Flatten
# ---------------------------------------------------------------------------

class TestNaryFlatten:
    def test_scalar_flattens(self):
        a, b, c, d, e = getScalars('a b c d e')
        result = Add(a, b, Add(c, d, e))
        assert result.arity == 5
        assert str(result) == '(a+b+c+d+e)'

    def test_vector_flattens(self):
        x, y, z = getVectors(['x', 'y', 'z'])
        result = VAdd(x, VAdd(y, z))
        assert result.arity == 3
        assert str(result) == '(x+y+z)'

    def test_matrix_flattens(self):
        A, B, C, D = getMatrices('A B C D')
        result = MAdd(A, MAdd(B, C, D))
        assert result.arity == 4
        assert str(result) == '(A+B+C+D)'

    def test_deep_flatten(self):
        a, b, c, d = getScalars('a b c d')
        result = Add(a, Add(b, Add(c, d)))
        assert result.arity == 4
        assert str(result) == '(a+b+c+d)'

    def test_list_arg(self):
        a, b, c = getScalars('a b c')
        result = Add([a, b], c)
        assert result.arity == 3
        assert str(result) == '(a+b+c)'

    def test_tuple_arg(self):
        x, y, z = getVectors(['x', 'y', 'z'])
        result = VAdd((x, y), z)
        assert result.arity == 3
        assert str(result) == '(x+y+z)'


# ---------------------------------------------------------------------------
# __str__
# ---------------------------------------------------------------------------

class TestAddStr:
    def test_scalar_constructor_str(self):
        a, b = getScalars('a b')
        assert str(Add(a, b)) == '(a+b)'

    def test_vector_constructor_str(self):
        x, y = getVectors(['x', 'y'])
        assert str(VAdd(x, y)) == '(x+y)'

    def test_matrix_constructor_str(self):
        M, N = getMatrices('M N')
        assert str(MAdd(M, N)) == '(M+N)'

    def test_scalar_operator_str(self):
        a, b = getScalars('a b')
        assert str(a + b) == '(a+b)'

    def test_vector_operator_str(self):
        x, y = getVectors(['x', 'y'])
        assert str(x + y) == '(x+y)'

    def test_matrix_operator_str(self):
        M, N = getMatrices('M N')
        assert str(M + N) == '(M+N)'


# ---------------------------------------------------------------------------
# __len__
# ---------------------------------------------------------------------------

class TestAddLen:
    def test_scalar_len(self):
        a, b, c = getScalars('a b c')
        assert len(Add(a, b, c)) == 3

    def test_vector_len(self):
        x, y = getVectors(['x', 'y'])
        assert len(VAdd(x, y)) == 2

    def test_matrix_len(self):
        A, B, C, D = getMatrices('A B C D')
        assert len(MAdd(A, B, C, D)) == 4


# ---------------------------------------------------------------------------
# __eq__ / __hash__ (string-based)
# ---------------------------------------------------------------------------

class TestAddEq:
    def test_scalar_eq(self):
        a, b = getScalars('a b')
        assert Add(a, b) == Add(a, b)

    def test_vector_eq(self):
        x, y = getVectors(['x', 'y'])
        assert VAdd(x, y) == VAdd(x, y)

    def test_matrix_eq(self):
        A, B = getMatrices('A B')
        assert MAdd(A, B) == MAdd(A, B)

    def test_not_eq_different_order(self):
        a, b = getScalars('a b')
        assert Add(a, b) != Add(b, a)

    def test_not_eq_different_arity(self):
        a, b, c = getScalars('a b c')
        assert Add(a, b) != Add(a, b, c)

    def test_hash_consistent(self):
        a, b = getScalars('a b')
        assert hash(Add(a, b)) == hash(Add(a, b))


# ---------------------------------------------------------------------------
# has()
# ---------------------------------------------------------------------------

class TestAddHas:
    def test_scalar_has_member(self):
        a, b, c = getScalars('a b c')
        result = Add(a, b)
        assert result.has(a)
        assert result.has(b)
        assert not result.has(c)

    def test_vector_has_member(self):
        x, y, z = getVectors(['x', 'y', 'z'])
        result = VAdd(x, y)
        assert result.has(x)
        assert not result.has(z)

    def test_matrix_has_member(self):
        A, B, C = getMatrices('A B C')
        result = MAdd(A, B)
        assert result.has(A)
        assert not result.has(C)

    def test_has_nested(self):
        a, b = getScalars('a b')
        x = Vector('x')
        from geomech.core.operations.multiplication import SVMul
        result = VAdd(SVMul(x, a), SVMul(x, b))
        assert result.has(x)
        assert result.has(a)
        assert result.has(b)


# ---------------------------------------------------------------------------
# Operator __add__ / __iadd__
# ---------------------------------------------------------------------------

class TestAddOperator:
    def test_scalar_add_returns_Add(self):
        a, b = getScalars('a b')
        result = a + b
        assert isinstance(result, Add)
        assert result.arity == 2

    def test_vector_add_returns_VAdd(self):
        x, y = getVectors(['x', 'y'])
        result = x + y
        assert isinstance(result, VAdd)
        assert result.arity == 2

    def test_matrix_add_returns_MAdd(self):
        M, N = getMatrices('M N')
        result = M + N
        assert isinstance(result, MAdd)
        assert result.arity == 2

    def test_scalar_iadd_flattens(self):
        a, b, c = getScalars('a b c')
        result = a + b
        result += c
        assert isinstance(result, Add)
        assert result.arity == 3
        assert str(result) == '(a+b+c)'

    def test_vector_iadd_flattens(self):
        x, y, z = getVectors(['x', 'y', 'z'])
        result = x + y
        result += z
        assert isinstance(result, VAdd)
        assert result.arity == 3
        assert str(result) == '(x+y+z)'

    def test_matrix_iadd_flattens(self):
        A, B, C = getMatrices('A B C')
        result = A + B
        result += C
        assert isinstance(result, MAdd)
        assert result.arity == 3
        assert str(result) == '(A+B+C)'


# ---------------------------------------------------------------------------
# Subtraction (__sub__)
# ---------------------------------------------------------------------------

class TestSubtraction:
    """a - b  =>  Add(a, SVMul(b, -1))  i.e.  a + b*(-1)"""

    def test_scalar_sub(self):
        a, b = getScalars('a b')
        result = a - b
        assert isinstance(result, Add)
        assert result.arity == 2
        assert str(result) == '(a+b(-1))'

    def test_vector_sub(self):
        x, y = getVectors(['x', 'y'])
        result = x - y
        assert isinstance(result, VAdd)
        assert result.arity == 2
        assert str(result) == '(x+y(-1))'

    def test_matrix_sub(self):
        M, N = getMatrices('M N')
        result = M - N
        assert isinstance(result, MAdd)
        assert result.arity == 2
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


# ---------------------------------------------------------------------------
# Type mismatch errors
# ---------------------------------------------------------------------------

class TestAddTypeMismatch:
    def test_scalar_vector_via_operator(self):
        a = Scalar('a')
        x = Vector('x')
        with pytest.raises(Exception):
            a + x

    def test_scalar_matrix_via_operator(self):
        a = Scalar('a')
        M = Matrix('M')
        with pytest.raises(Exception):
            a + M

    def test_vector_matrix_via_constructor(self):
        x = Vector('x')
        M = Matrix('M')
        with pytest.raises(ExpressionMismatchError):
            VAdd(x, M)

    def test_scalar_in_vadd(self):
        a = Scalar('a')
        x = Vector('x')
        with pytest.raises(ExpressionMismatchError):
            VAdd(a, x)

    def test_vector_in_madd(self):
        x = Vector('x')
        A = Matrix('A')
        with pytest.raises(ExpressionMismatchError):
            MAdd(x, A)

    def test_list_with_wrong_type(self):
        a = Scalar('a')
        x, y = getVectors(['x', 'y'])
        with pytest.raises(ExpressionMismatchError):
            VAdd([a, x], y)


# ---------------------------------------------------------------------------
# Size mismatch errors
# ---------------------------------------------------------------------------

class TestAddSizeMismatch:
    def test_vector_size_mismatch(self):
        x = Vector('x', size=(3,))
        y = Vector('y', size=(4,))
        with pytest.raises(SizeMismatchError):
            VAdd(x, y)

    def test_matrix_size_mismatch(self):
        A = Matrix('A', size=(3, 3))
        B = Matrix('B', size=(4, 4))
        with pytest.raises(SizeMismatchError):
            MAdd(A, B)

    def test_vector_size_mismatch_in_list(self):
        x = Vector('x', size=(3,))
        y = Vector('y', size=(4,))
        with pytest.raises(SizeMismatchError):
            VAdd([x, y])

    def test_same_size_passes(self):
        x = Vector('x', size=(3,))
        y = Vector('y', size=(3,))
        result = VAdd(x, y)
        assert result.arity == 2


# ---------------------------------------------------------------------------
# delta / diff on Add (Phase 2 — not yet implemented)
# ---------------------------------------------------------------------------

class TestAdditionDelta:
    def test_scalar_add_delta(self):
        a, b = getScalars('a b')
        d = (a + b).delta()
        assert isinstance(d, Add)
        assert d.arity == 2
        assert str(d) == '(\\delta{a}+\\delta{b})'

    def test_vector_add_delta(self):
        x, y = getVectors(['x', 'y'])
        d = (x + y).delta()
        assert isinstance(d, VAdd)
        assert d.arity == 2
        assert str(d) == '(\\delta{x}+\\delta{y})'

    def test_matrix_add_delta(self):
        M, N = getMatrices('M N')
        d = (M + N).delta()
        assert isinstance(d, MAdd)
        assert d.arity == 2
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
        d = (a + b).t_diff()
        assert isinstance(d, Add)
        assert str(d) == '(\\frac{d}{dt}(a)+\\frac{d}{dt}(b))'

    def test_vector_add_diff(self):
        x, y = getVectors(['x', 'y'])
        d = (x + y).t_diff()
        assert isinstance(d, VAdd)
        assert str(d) == '(\\frac{d}{dt}(x)+\\frac{d}{dt}(y))'

    def test_matrix_add_diff(self):
        M, N = getMatrices('M N')
        d = (M + N).t_diff()
        assert isinstance(d, MAdd)
        assert str(d) == '(\\frac{d}{dt}(M)+\\frac{d}{dt}(N))'

    def test_all_constant_scalar_diff_is_zero(self):
        m, g = getScalars('m g', attr=['Constant'])
        d = (m + g).t_diff()
        assert d == Zero

    def test_all_constant_vector_diff_is_zero(self):
        e1 = Vector('e1', attr=['Constant'])
        e2 = Vector('e2', attr=['Constant'])
        d = (e1 + e2).t_diff()
        assert d == ZeroVector

    def test_all_constant_matrix_diff_is_zero(self):
        A = Matrix('A', attr=['Constant'])
        B = Matrix('B', attr=['Constant'])
        d = (A + B).t_diff()
        assert d == ZeroMatrix
