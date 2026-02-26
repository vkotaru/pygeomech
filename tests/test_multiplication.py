import pytest
from geomech.core.expressions import (
    Scalar, Vector, Matrix,
    getScalars, getVectors, getMatrices,
)
from geomech.core.operations.multiplication import (
    Mul, SVMul, SMMul, MVMul, MMMul, VVMul,
)
from geomech.core.operations.addition import Add, VAdd, MAdd
from geomech.core.operations.geometry import Transpose
from geomech.core.types import ExprType
from geomech.utils.errors import ExpressionMismatchError, SizeMismatchError


# ---------------------------------------------------------------------------
# Type
# ---------------------------------------------------------------------------

class TestMulType:
    def test_mul_type(self):
        a, b = getScalars('a b')
        assert Mul(a, b).type == ExprType.SCALAR

    def test_svmul_type(self):
        x = Vector('x')
        a = Scalar('a')
        assert SVMul(x, a).type == ExprType.VECTOR

    def test_smmul_type(self):
        M = Matrix('M')
        a = Scalar('a')
        assert SMMul(M, a).type == ExprType.MATRIX

    def test_mvmul_type(self):
        M = Matrix('M')
        x = Vector('x')
        assert MVMul(M, x).type == ExprType.VECTOR

    def test_mmmul_type(self):
        M, N = getMatrices('M N')
        assert MMMul(M, N).type == ExprType.MATRIX

    def test_vvmul_inner_type(self):
        x, y = getVectors(['x', 'y'])
        assert VVMul(Transpose(x), y).type == ExprType.SCALAR

    def test_vvmul_outer_type(self):
        x, y = getVectors(['x', 'y'])
        assert VVMul(x, Transpose(y)).type == ExprType.MATRIX


# ---------------------------------------------------------------------------
# Construction & arity & nodes
# ---------------------------------------------------------------------------

class TestMulConstruction:
    def test_mul_nodes(self):
        a, b = getScalars('a b')
        result = Mul(a, b)
        assert isinstance(result, Mul)
        assert result.arity == 2
        assert result.left == a
        assert result.right == b

    def test_svmul_nodes(self):
        x = Vector('x')
        a = Scalar('a')
        result = SVMul(x, a)
        assert isinstance(result, SVMul)
        assert result.arity == 2
        assert result.left == x
        assert result.right == a

    def test_smmul_nodes(self):
        M = Matrix('M')
        a = Scalar('a')
        result = SMMul(M, a)
        assert isinstance(result, SMMul)
        assert result.arity == 2
        assert result.left == M
        assert result.right == a

    def test_mvmul_nodes(self):
        M = Matrix('M')
        x = Vector('x')
        result = MVMul(M, x)
        assert isinstance(result, MVMul)
        assert result.arity == 2
        assert result.left == M
        assert result.right == x

    def test_mmmul_nodes(self):
        M, N = getMatrices('M N')
        result = MMMul(M, N)
        assert isinstance(result, MMMul)
        assert result.arity == 2
        assert result.left == M
        assert result.right == N

    def test_vvmul_inner_nodes(self):
        x, y = getVectors(['x', 'y'])
        tx = Transpose(x)
        result = VVMul(tx, y)
        assert isinstance(result, VVMul)
        assert result.arity == 2
        assert result.left == tx
        assert result.right == y

    def test_vvmul_outer_nodes(self):
        x, y = getVectors(['x', 'y'])
        ty = Transpose(y)
        result = VVMul(x, ty)
        assert result.arity == 2
        assert result.left == x
        assert result.right == ty


# ---------------------------------------------------------------------------
# Numeric coercion
# ---------------------------------------------------------------------------

class TestNumericCoercion:
    def test_mul_int_right(self):
        a = Scalar('a')
        result = Mul(a, 2)
        assert isinstance(result, Mul)
        assert str(result) == 'a(2)'

    def test_mul_int_left(self):
        a = Scalar('a')
        result = Mul(2, a)
        assert isinstance(result, Mul)
        assert str(result) == '(2)a'

    def test_mul_float(self):
        a = Scalar('a')
        result = Mul(a, 0.5)
        assert isinstance(result, Mul)
        assert str(result) == 'a(0.5)'

    def test_svmul_int(self):
        x = Vector('x')
        result = SVMul(x, 3)
        assert isinstance(result, SVMul)
        assert str(result) == 'x(3)'

    def test_svmul_int_left(self):
        x = Vector('x')
        result = SVMul(3, x)
        assert isinstance(result, SVMul)
        assert result.left == x
        assert str(result) == 'x(3)'

    def test_smmul_int(self):
        M = Matrix('M')
        result = SMMul(M, 2)
        assert isinstance(result, SMMul)
        assert str(result) == 'M(2)'

    def test_smmul_int_left(self):
        M = Matrix('M')
        result = SMMul(2, M)
        assert isinstance(result, SMMul)
        assert result.left == M
        assert str(result) == 'M(2)'


# ---------------------------------------------------------------------------
# Normalization (SVMul, SMMul always store vector/matrix left, scalar right)
# ---------------------------------------------------------------------------

class TestNormalization:
    def test_svmul_vector_scalar(self):
        x = Vector('x')
        a = Scalar('a')
        result = SVMul(x, a)
        assert result.left == x
        assert result.right == a

    def test_svmul_scalar_vector(self):
        x = Vector('x')
        a = Scalar('a')
        result = SVMul(a, x)
        assert result.left == x
        assert result.right == a

    def test_smmul_matrix_scalar(self):
        M = Matrix('M')
        a = Scalar('a')
        result = SMMul(M, a)
        assert result.left == M
        assert result.right == a

    def test_smmul_scalar_matrix(self):
        M = Matrix('M')
        a = Scalar('a')
        result = SMMul(a, M)
        assert result.left == M
        assert result.right == a


# ---------------------------------------------------------------------------
# __str__
# ---------------------------------------------------------------------------

class TestMulStr:
    def test_mul_str(self):
        a, b = getScalars('a b')
        assert str(Mul(a, b)) == 'ab'

    def test_svmul_str(self):
        x = Vector('x')
        a = Scalar('a')
        assert str(SVMul(x, a)) == 'xa'

    def test_smmul_str(self):
        M = Matrix('M')
        a = Scalar('a')
        assert str(SMMul(M, a)) == 'Ma'

    def test_mvmul_str(self):
        M = Matrix('M')
        x = Vector('x')
        assert str(MVMul(M, x)) == 'Mx'

    def test_mmmul_str(self):
        M, N = getMatrices('M N')
        assert str(MMMul(M, N)) == 'MN'

    def test_vvmul_inner_str(self):
        x, y = getVectors(['x', 'y'])
        assert str(VVMul(Transpose(x), y)) == "(x)'y"

    def test_vvmul_outer_str(self):
        x, y = getVectors(['x', 'y'])
        assert str(VVMul(x, Transpose(y))) == "x(y)'"


# ---------------------------------------------------------------------------
# __len__
# ---------------------------------------------------------------------------

class TestMulLen:
    def test_mul_len(self):
        a, b = getScalars('a b')
        assert len(Mul(a, b)) == 2

    def test_svmul_len(self):
        x = Vector('x')
        a = Scalar('a')
        assert len(SVMul(x, a)) == 2

    def test_mvmul_len(self):
        M = Matrix('M')
        x = Vector('x')
        assert len(MVMul(M, x)) == 2

    def test_mmmul_len(self):
        M, N = getMatrices('M N')
        assert len(MMMul(M, N)) == 2

    def test_smmul_len(self):
        M = Matrix('M')
        a = Scalar('a')
        assert len(SMMul(M, a)) == 2

    def test_vvmul_len(self):
        x, y = getVectors(['x', 'y'])
        assert len(VVMul(Transpose(x), y)) == 2


# ---------------------------------------------------------------------------
# __eq__ / __hash__ (string-based)
# ---------------------------------------------------------------------------

class TestMulEq:
    def test_mul_eq(self):
        a, b = getScalars('a b')
        assert Mul(a, b) == Mul(a, b)

    def test_mul_ne_order(self):
        a, b = getScalars('a b')
        assert Mul(a, b) != Mul(b, a)

    def test_svmul_eq(self):
        x = Vector('x')
        a = Scalar('a')
        assert SVMul(x, a) == SVMul(x, a)

    def test_mvmul_eq(self):
        M = Matrix('M')
        x = Vector('x')
        assert MVMul(M, x) == MVMul(M, x)

    def test_mmmul_eq(self):
        M, N = getMatrices('M N')
        assert MMMul(M, N) == MMMul(M, N)

    def test_smmul_eq(self):
        M = Matrix('M')
        a = Scalar('a')
        assert SMMul(M, a) == SMMul(M, a)

    def test_vvmul_eq(self):
        x, y = getVectors(['x', 'y'])
        assert VVMul(Transpose(x), y) == VVMul(Transpose(x), y)

    def test_hash_consistent(self):
        a, b = getScalars('a b')
        assert hash(Mul(a, b)) == hash(Mul(a, b))


# ---------------------------------------------------------------------------
# Constant / Zero propagation
# ---------------------------------------------------------------------------

class TestConstantZeroPropagation:
    def test_mul_constant(self):
        m = Scalar('m', attr=['Constant'])
        g = Scalar('g', attr=['Constant'])
        assert Mul(m, g).isConstant

    def test_mul_not_constant(self):
        a, b = getScalars('a b')
        assert not Mul(a, b).isConstant

    def test_mul_zero_left(self):
        z = Scalar('0', attr=['Constant', 'Zero'])
        a = Scalar('a')
        assert Mul(z, a).isZero

    def test_mul_zero_right(self):
        a = Scalar('a')
        z = Scalar('0', attr=['Constant', 'Zero'])
        assert Mul(a, z).isZero

    def test_svmul_constant(self):
        m = Scalar('m', attr=['Constant'])
        e = Vector('e', attr=['Constant'])
        assert SVMul(e, m).isConstant

    def test_svmul_zero_scalar(self):
        z = Scalar('0', attr=['Constant', 'Zero'])
        x = Vector('x')
        assert SVMul(x, z).isZero

    def test_mvmul_constant(self):
        J = Matrix('J', attr=['Constant'])
        e = Vector('e', attr=['Constant'])
        assert MVMul(J, e).isConstant

    def test_smmul_constant(self):
        m = Scalar('m', attr=['Constant'])
        J = Matrix('J', attr=['Constant'])
        assert SMMul(J, m).isConstant

    def test_smmul_zero(self):
        z = Scalar('0', attr=['Constant', 'Zero'])
        M = Matrix('M')
        assert SMMul(M, z).isZero

    def test_mvmul_zero(self):
        M = Matrix('M', attr=['Constant', 'Zero'])
        x = Vector('x')
        assert MVMul(M, x).isZero

    def test_mmmul_constant(self):
        J = Matrix('J', attr=['Constant'])
        K = Matrix('K', attr=['Constant'])
        assert MMMul(J, K).isConstant

    def test_mmmul_zero(self):
        Z = Matrix('Z', attr=['Constant', 'Zero'])
        M = Matrix('M')
        assert MMMul(Z, M).isZero


# ---------------------------------------------------------------------------
# has()
# ---------------------------------------------------------------------------

class TestMulHas:
    def test_mul_has_left(self):
        a, b, c = getScalars('a b c')
        result = Mul(a, b)
        assert result.has(a)
        assert result.has(b)
        assert not result.has(c)

    def test_svmul_has(self):
        x = Vector('x')
        a = Scalar('a')
        result = SVMul(x, a)
        assert result.has(x)
        assert result.has(a)

    def test_mvmul_has(self):
        M = Matrix('M')
        x = Vector('x')
        result = MVMul(M, x)
        assert result.has(M)
        assert result.has(x)

    def test_has_nested(self):
        a, b = getScalars('a b')
        x = Vector('x')
        result = SVMul(x, Mul(a, b))
        assert result.has(x)
        assert result.has(a)
        assert result.has(b)


# ---------------------------------------------------------------------------
# Operator __mul__
# ---------------------------------------------------------------------------

class TestMulOperator:
    def test_scalar_times_scalar(self):
        a, b = getScalars('a b')
        result = a * b
        assert isinstance(result, Mul)

    def test_scalar_times_vector(self):
        a = Scalar('a')
        x = Vector('x')
        result = a * x
        assert isinstance(result, SVMul)

    def test_scalar_times_matrix(self):
        a = Scalar('a')
        M = Matrix('M')
        result = a * M
        assert isinstance(result, SMMul)

    def test_vector_times_scalar(self):
        x = Vector('x')
        a = Scalar('a')
        result = x * a
        assert isinstance(result, SVMul)

    def test_vector_times_int(self):
        x = Vector('x')
        result = x * 3
        assert isinstance(result, SVMul)
        assert str(result) == 'x(3)'

    def test_matrix_times_vector(self):
        M = Matrix('M')
        x = Vector('x')
        result = M * x
        assert isinstance(result, MVMul)

    def test_matrix_times_matrix(self):
        M, N = getMatrices('M N')
        result = M * N
        assert isinstance(result, MMMul)

    def test_matrix_times_scalar(self):
        M = Matrix('M')
        a = Scalar('a')
        result = M * a
        assert isinstance(result, SMMul)

    def test_matrix_times_int(self):
        M = Matrix('M')
        result = M * 2
        assert isinstance(result, SMMul)
        assert str(result) == 'M(2)'

    def test_transposeT_times_vector(self):
        x, y = getVectors(['x', 'y'])
        result = Transpose(x) * y
        assert isinstance(result, VVMul)
        assert result.type == ExprType.SCALAR


# ---------------------------------------------------------------------------
# VVMul __add__ / __mul__ dispatch
# ---------------------------------------------------------------------------

class TestVVMulDispatch:
    def test_inner_add_scalar(self):
        x, y = getVectors(['x', 'y'])
        a = Scalar('a')
        result = VVMul(Transpose(x), y) + a
        assert isinstance(result, Add)

    def test_outer_add_matrix(self):
        x, y = getVectors(['x', 'y'])
        M = Matrix('M')
        result = VVMul(x, Transpose(y)) + M
        assert isinstance(result, MAdd)

    def test_inner_mul_scalar(self):
        x, y = getVectors(['x', 'y'])
        a = Scalar('a')
        result = VVMul(Transpose(x), y) * a
        assert isinstance(result, Mul)

    def test_inner_mul_vector(self):
        x, y, z = getVectors(['x', 'y', 'z'])
        result = VVMul(Transpose(x), y) * z
        assert isinstance(result, SVMul)

    def test_outer_mul_scalar(self):
        x, y = getVectors(['x', 'y'])
        a = Scalar('a')
        result = VVMul(x, Transpose(y)) * a
        assert isinstance(result, SMMul)

    def test_outer_mul_vector(self):
        x, y, z = getVectors(['x', 'y', 'z'])
        result = VVMul(x, Transpose(y)) * z
        assert isinstance(result, MVMul)

    def test_outer_mul_matrix(self):
        x, y = getVectors(['x', 'y'])
        M = Matrix('M')
        result = VVMul(x, Transpose(y)) * M
        assert isinstance(result, MMMul)


# ---------------------------------------------------------------------------
# Type mismatch errors
# ---------------------------------------------------------------------------

class TestMulTypeMismatch:
    def test_mul_rejects_vector(self):
        a = Scalar('a')
        x = Vector('x')
        with pytest.raises(ExpressionMismatchError):
            Mul(a, x)

    def test_mul_rejects_matrix(self):
        a = Scalar('a')
        M = Matrix('M')
        with pytest.raises(ExpressionMismatchError):
            Mul(a, M)

    def test_svmul_rejects_two_scalars(self):
        a, b = getScalars('a b')
        with pytest.raises(ExpressionMismatchError):
            SVMul(a, b)

    def test_svmul_rejects_matrix(self):
        x = Vector('x')
        M = Matrix('M')
        with pytest.raises(ExpressionMismatchError):
            SVMul(x, M)

    def test_smmul_rejects_two_scalars(self):
        a, b = getScalars('a b')
        with pytest.raises(ExpressionMismatchError):
            SMMul(a, b)

    def test_smmul_rejects_vector(self):
        M = Matrix('M')
        x = Vector('x')
        with pytest.raises(ExpressionMismatchError):
            SMMul(M, x)

    def test_mvmul_rejects_two_scalars(self):
        a, b = getScalars('a b')
        with pytest.raises(ExpressionMismatchError):
            MVMul(a, b)

    def test_mvmul_rejects_vec_mat_no_transpose(self):
        x = Vector('x')
        M = Matrix('M')
        with pytest.raises(ExpressionMismatchError):
            MVMul(x, M)

    def test_mmmul_rejects_scalar(self):
        M = Matrix('M')
        a = Scalar('a')
        with pytest.raises(ExpressionMismatchError):
            MMMul(M, a)

    def test_mmmul_rejects_vector(self):
        M = Matrix('M')
        x = Vector('x')
        with pytest.raises(ExpressionMismatchError):
            MMMul(M, x)

    def test_vvmul_rejects_both_raw(self):
        x, y = getVectors(['x', 'y'])
        with pytest.raises(ExpressionMismatchError):
            VVMul(x, y)

    def test_vvmul_rejects_both_transposed(self):
        x, y = getVectors(['x', 'y'])
        with pytest.raises(ExpressionMismatchError):
            VVMul(Transpose(x), Transpose(y))

    def test_vvmul_rejects_scalar_vector(self):
        a = Scalar('a')
        x = Vector('x')
        with pytest.raises(ExpressionMismatchError):
            VVMul(a, x)


# ---------------------------------------------------------------------------
# Size mismatch errors
# ---------------------------------------------------------------------------

class TestMulSizeMismatch:
    def test_mvmul_size_mismatch(self):
        M = Matrix('M', size=(3, 4))
        x = Vector('x', size=(3,))
        with pytest.raises(SizeMismatchError):
            MVMul(M, x)

    def test_mvmul_size_ok(self):
        M = Matrix('M', size=(3, 4))
        x = Vector('x', size=(4,))
        result = MVMul(M, x)
        assert result.arity == 2

    def test_mmmul_size_mismatch(self):
        A = Matrix('A', size=(3, 4))
        B = Matrix('B', size=(5, 2))
        with pytest.raises(SizeMismatchError):
            MMMul(A, B)

    def test_mmmul_size_ok(self):
        A = Matrix('A', size=(3, 4))
        B = Matrix('B', size=(4, 2))
        result = MMMul(A, B)
        assert result.arity == 2

    def test_vvmul_inner_size_mismatch(self):
        x = Vector('x', size=(3,))
        y = Vector('y', size=(4,))
        with pytest.raises(SizeMismatchError):
            VVMul(Transpose(x), y)

    def test_vvmul_inner_size_ok(self):
        x = Vector('x', size=(3,))
        y = Vector('y', size=(3,))
        result = VVMul(Transpose(x), y)
        assert result.type == ExprType.SCALAR

    def test_vvmul_outer_any_sizes(self):
        x = Vector('x', size=(3,))
        y = Vector('y', size=(5,))
        result = VVMul(x, Transpose(y))
        assert result.type == ExprType.MATRIX

    def test_mvmul_no_size_skips(self):
        """When operands have no .size (compound expressions), no error."""
        M = Matrix('M')
        x = Vector('x')
        a = Scalar('a')
        # SVMul(x, a) has no .size attribute — should not raise
        result = MVMul(M, SVMul(x, a))
        assert result.arity == 2


# ---------------------------------------------------------------------------
# delta / diff on multiplication (Phase 2 — not yet implemented)
# ---------------------------------------------------------------------------

@pytest.mark.xfail(reason="delta/diff on operations not yet implemented (Phase 2)")
class TestMulDelta:
    def test_mul_delta_both_variable(self):
        a, b = getScalars('a b')
        d = Mul(a, b).delta()
        assert isinstance(d, Add)
        assert d.arity == 2
        assert str(d) == '(\\delta{a}b+a\\delta{b})'

    def test_mul_delta_constant_left(self):
        m = Scalar('m', attr=['Constant'])
        a = Scalar('a')
        d = Mul(m, a).delta()
        assert isinstance(d, Mul)
        assert str(d) == 'm\\delta{a}'

    def test_mul_delta_constant_right(self):
        a = Scalar('a')
        m = Scalar('m', attr=['Constant'])
        d = Mul(a, m).delta()
        assert isinstance(d, Mul)
        assert str(d) == '\\delta{a}m'

    def test_mul_delta_both_constant(self):
        m = Scalar('m', attr=['Constant'])
        g = Scalar('g', attr=['Constant'])
        d = Mul(m, g).delta()
        assert d.isZero

    def test_svmul_delta(self):
        a = Scalar('a')
        x = Vector('x')
        d = SVMul(x, a).delta()
        assert isinstance(d, VAdd)
        assert d.arity == 2
        assert str(d) == '(\\delta{x}a+x\\delta{a})'

    def test_svmul_delta_constant_scalar(self):
        m = Scalar('m', attr=['Constant'])
        x = Vector('x')
        d = SVMul(x, m).delta()
        assert isinstance(d, SVMul)
        assert str(d) == '\\delta{x}m'

    def test_smmul_delta(self):
        a = Scalar('a')
        M = Matrix('M')
        d = SMMul(M, a).delta()
        assert isinstance(d, MAdd)
        assert d.arity == 2

    def test_mvmul_delta(self):
        M = Matrix('M')
        x = Vector('x')
        d = MVMul(M, x).delta()
        assert isinstance(d, VAdd)
        assert d.arity == 2
        assert str(d) == '(\\delta{M}x+M\\delta{x})'

    def test_mvmul_delta_constant_matrix(self):
        J = Matrix('J', attr=['Constant'])
        x = Vector('x')
        d = MVMul(J, x).delta()
        assert isinstance(d, MVMul)
        assert d.left == J

    def test_mmmul_delta(self):
        M, N = getMatrices('M N')
        d = MMMul(M, N).delta()
        assert isinstance(d, MAdd)
        assert d.arity == 2
        assert str(d) == '(\\delta{M}N+M\\delta{N})'

    def test_vvmul_inner_delta(self):
        x, y = getVectors(['x', 'y'])
        d = VVMul(Transpose(x), y).delta()
        assert isinstance(d, Add)
        assert d.arity == 2
        assert str(d) == "((\\delta{x})'y+(x)'\\delta{y})"

    def test_vvmul_outer_delta(self):
        x, y = getVectors(['x', 'y'])
        d = VVMul(x, Transpose(y)).delta()
        assert isinstance(d, MAdd)
        assert d.arity == 2
        assert str(d) == "(\\delta{x}(y)'+x(\\delta{y})')"


@pytest.mark.xfail(reason="delta/diff on operations not yet implemented (Phase 2)")
class TestMulDiff:
    def test_mul_diff_both_variable(self):
        a, b = getScalars('a b')
        d = Mul(a, b).diff()
        assert isinstance(d, Add)
        assert str(d) == '(dot_ab+adot_b)'

    def test_mul_diff_constant_left(self):
        m = Scalar('m', attr=['Constant'])
        a = Scalar('a')
        d = Mul(m, a).diff()
        assert isinstance(d, Mul)
        assert str(d) == 'mdot_a'

    def test_mul_diff_both_constant(self):
        m = Scalar('m', attr=['Constant'])
        g = Scalar('g', attr=['Constant'])
        d = Mul(m, g).diff()
        assert d.isZero

    def test_mvmul_diff(self):
        M = Matrix('M')
        x = Vector('x')
        d = MVMul(M, x).diff()
        assert isinstance(d, VAdd)
        assert d.arity == 2

    def test_mmmul_diff(self):
        M, N = getMatrices('M N')
        d = MMMul(M, N).diff()
        assert isinstance(d, MAdd)
        assert d.arity == 2
