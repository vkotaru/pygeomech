import pytest
from geomech.core.expressions import (
    Scalar, Vector, Matrix,
    getScalars, getVectors, getMatrices,
)
from geomech.core.operations.addition import Add, VAdd, MAdd
from geomech.core.operations.multiplication import (
    Mul, SVMul, SMMul, MVMul,
)
from geomech.core.operations.calculus import Variation, TimeDerivative, TimeIntegral
from geomech.core.types import ExprType
from geomech.utils.errors import ExpressionMismatchError


# ===================================================================
# Variation
# ===================================================================

class TestVariationType:
    def test_scalar(self):
        a = Scalar('a')
        assert Variation(a).type == ExprType.SCALAR

    def test_vector(self):
        x = Vector('x')
        assert Variation(x).type == ExprType.VECTOR

    def test_matrix(self):
        M = Matrix('M')
        assert Variation(M).type == ExprType.MATRIX


class TestVariationConstruction:
    def test_expr(self):
        x = Vector('x')
        v = Variation(x)
        assert v.expr == x

    def test_nodes(self):
        a = Scalar('a')
        assert Variation(a).nodes == [a]

    def test_arity(self):
        x = Vector('x')
        assert Variation(x).arity == 1

    def test_len(self):
        x = Vector('x')
        assert len(Variation(x)) == 1


class TestVariationStr:
    def test_scalar(self):
        a = Scalar('a')
        assert str(Variation(a)) == '\\delta{a}'

    def test_vector(self):
        x = Vector('x')
        assert str(Variation(x)) == '\\delta{x}'

    def test_matrix(self):
        M = Matrix('M')
        assert str(Variation(M)) == '\\delta{M}'


class TestVariationProperties:
    def test_is_constant_true(self):
        c = Scalar('c', attr=['Constant'])
        assert Variation(c).isConstant is True

    def test_is_constant_false(self):
        a = Scalar('a')
        assert Variation(a).isConstant is False

    def test_is_zero_true(self):
        z = Vector('0', attr=['Constant', 'Zero'])
        assert Variation(z).isZero is True

    def test_is_zero_false(self):
        x = Vector('x')
        assert Variation(x).isZero is False


class TestVariationHas:
    def test_has_inner(self):
        x = Vector('x')
        assert Variation(x).has(x) is True

    def test_has_missing(self):
        x = Vector('x')
        y = Vector('y')
        assert Variation(x).has(y) is False


class TestVariationEqHash:
    def test_eq(self):
        x = Vector('x')
        assert Variation(x) == Variation(x)

    def test_neq(self):
        x = Vector('x')
        y = Vector('y')
        assert Variation(x) != Variation(y)

    def test_hash(self):
        x = Vector('x')
        assert hash(Variation(x)) == hash(Variation(x))

    def test_set(self):
        x = Vector('x')
        assert len({Variation(x), Variation(x)}) == 1


class TestVariationOperators:
    def test_add_scalar(self):
        a, b = getScalars('a b')
        result = Variation(a) + b
        assert isinstance(result, Add)

    def test_add_vector(self):
        x, y = getVectors(['x', 'y'])
        result = Variation(x) + y
        assert isinstance(result, VAdd)

    def test_add_matrix(self):
        M, N = getMatrices('M N')
        result = Variation(M) + N
        assert isinstance(result, MAdd)

    def test_sub_scalar(self):
        a, b = getScalars('a b')
        result = Variation(a) - b
        assert isinstance(result, Add)

    def test_sub_vector(self):
        x, y = getVectors(['x', 'y'])
        result = Variation(x) - y
        assert isinstance(result, VAdd)

    def test_sub_matrix(self):
        M, N = getMatrices('M N')
        result = Variation(M) - N
        assert isinstance(result, MAdd)

    def test_mul_scalar_scalar(self):
        a, b = getScalars('a b')
        result = Variation(a) * b
        assert isinstance(result, Mul)

    def test_mul_scalar_vector(self):
        a = Scalar('a')
        x = Vector('x')
        result = Variation(a) * x
        assert isinstance(result, SVMul)

    def test_mul_scalar_matrix(self):
        a = Scalar('a')
        M = Matrix('M')
        result = Variation(a) * M
        assert isinstance(result, SMMul)

    def test_mul_vector_scalar(self):
        x = Vector('x')
        a = Scalar('a')
        result = Variation(x) * a
        assert isinstance(result, SVMul)

    def test_mul_matrix_vector(self):
        M = Matrix('M')
        x = Vector('x')
        result = Variation(M) * x
        assert isinstance(result, MVMul)

    def test_mul_numeric(self):
        a = Scalar('a')
        result = Variation(a) * 3
        assert isinstance(result, Mul)

    def test_iadd(self):
        a, b = getScalars('a b')
        result = Variation(a)
        result += b
        assert isinstance(result, Add)

    def test_sub_type_mismatch(self):
        a = Scalar('a')
        x = Vector('x')
        with pytest.raises(ExpressionMismatchError):
            Variation(a) - x


# ===================================================================
# TimeDerivative
# ===================================================================

class TestTimeDerivativeType:
    def test_scalar(self):
        a = Scalar('a')
        assert TimeDerivative(a).type == ExprType.SCALAR

    def test_vector(self):
        x = Vector('x')
        assert TimeDerivative(x).type == ExprType.VECTOR

    def test_matrix(self):
        M = Matrix('M')
        assert TimeDerivative(M).type == ExprType.MATRIX


class TestTimeDerivativeConstruction:
    def test_expr(self):
        x = Vector('x')
        assert TimeDerivative(x).expr == x

    def test_arity(self):
        x = Vector('x')
        assert TimeDerivative(x).arity == 1

    def test_len(self):
        x = Vector('x')
        assert len(TimeDerivative(x)) == 1


class TestTimeDerivativeStr:
    def test_scalar(self):
        a = Scalar('a')
        assert str(TimeDerivative(a)) == '\\frac{d}{dt}(a)'

    def test_vector(self):
        x = Vector('x')
        assert str(TimeDerivative(x)) == '\\frac{d}{dt}(x)'

    def test_matrix(self):
        M = Matrix('M')
        assert str(TimeDerivative(M)) == '\\frac{d}{dt}(M)'


class TestTimeDerivativeProperties:
    def test_is_constant_true(self):
        c = Scalar('c', attr=['Constant'])
        assert TimeDerivative(c).isConstant is True

    def test_is_constant_false(self):
        a = Scalar('a')
        assert TimeDerivative(a).isConstant is False

    def test_is_zero_true(self):
        z = Vector('0', attr=['Constant', 'Zero'])
        assert TimeDerivative(z).isZero is True

    def test_is_zero_false(self):
        x = Vector('x')
        assert TimeDerivative(x).isZero is False


class TestTimeDerivativeOperators:
    def test_add_scalar(self):
        a, b = getScalars('a b')
        result = TimeDerivative(a) + b
        assert isinstance(result, Add)

    def test_sub_vector(self):
        x, y = getVectors(['x', 'y'])
        result = TimeDerivative(x) - y
        assert isinstance(result, VAdd)

    def test_mul_scalar(self):
        a, b = getScalars('a b')
        result = TimeDerivative(a) * b
        assert isinstance(result, Mul)

    def test_mul_numeric(self):
        x = Vector('x')
        result = TimeDerivative(x) * 2
        assert isinstance(result, SVMul)


# ===================================================================
# TimeIntegral
# ===================================================================

class TestTimeIntegralType:
    def test_scalar(self):
        a = Scalar('a')
        assert TimeIntegral(a).type == ExprType.SCALAR

    def test_vector(self):
        x = Vector('x')
        assert TimeIntegral(x).type == ExprType.VECTOR

    def test_matrix(self):
        M = Matrix('M')
        assert TimeIntegral(M).type == ExprType.MATRIX


class TestTimeIntegralConstruction:
    def test_expr(self):
        x = Vector('x')
        assert TimeIntegral(x).expr == x

    def test_arity(self):
        x = Vector('x')
        assert TimeIntegral(x).arity == 1

    def test_len(self):
        x = Vector('x')
        assert len(TimeIntegral(x)) == 1


class TestTimeIntegralStr:
    def test_scalar(self):
        a = Scalar('a')
        assert str(TimeIntegral(a)) == '\\int{a}dt'

    def test_vector(self):
        x = Vector('x')
        assert str(TimeIntegral(x)) == '\\int{x}dt'

    def test_matrix(self):
        M = Matrix('M')
        assert str(TimeIntegral(M)) == '\\int{M}dt'


class TestTimeIntegralProperties:
    def test_is_constant_true(self):
        c = Scalar('c', attr=['Constant'])
        assert TimeIntegral(c).isConstant is True

    def test_is_constant_false(self):
        a = Scalar('a')
        assert TimeIntegral(a).isConstant is False

    def test_is_zero_true(self):
        z = Vector('0', attr=['Constant', 'Zero'])
        assert TimeIntegral(z).isZero is True

    def test_is_zero_false(self):
        x = Vector('x')
        assert TimeIntegral(x).isZero is False


class TestTimeIntegralOperators:
    def test_add_scalar(self):
        a, b = getScalars('a b')
        result = TimeIntegral(a) + b
        assert isinstance(result, Add)

    def test_sub_matrix(self):
        M, N = getMatrices('M N')
        result = TimeIntegral(M) - N
        assert isinstance(result, MAdd)

    def test_mul_scalar(self):
        a, b = getScalars('a b')
        result = TimeIntegral(a) * b
        assert isinstance(result, Mul)

    def test_mul_matrix_vector(self):
        M = Matrix('M')
        x = Vector('x')
        result = TimeIntegral(M) * x
        assert isinstance(result, MVMul)
