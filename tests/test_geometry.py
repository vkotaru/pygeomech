import pytest

from geomech.core.base.expressions import (
    Matrix,
    Scalar,
    Vector,
    getMatrices,
    getScalars,
    getVectors,
)
from geomech.core.base.types import ExprType
from geomech.core.operations.addition import Add, MAdd, VAdd
from geomech.core.operations.calculus import Variation
from geomech.core.operations.geometry import Cross, Dot, Hat, Transpose, Vee
from geomech.core.operations.multiplication import (
    Mul,
    MVMul,
    SVMul,
    VVMul,
)
from geomech.utils.errors import ExpressionMismatchError

# ===================================================================
# Variation (TestVariationType is in test_calculus.py)
# ===================================================================


class TestVariationConstruction:
    def test_expr(self):
        x = Vector("x")
        d = Variation(x)
        assert d.expr == x

    def test_nodes(self):
        a = Scalar("a")
        d = Variation(a)
        assert d.nodes == [a]

    def test_arity(self):
        x = Vector("x")
        assert Variation(x).arity == 1

    def test_len(self):
        x = Vector("x")
        assert len(Variation(x)) == 1


class TestVariationStr:
    def test_scalar(self):
        a = Scalar("a")
        assert str(Variation(a)) == "\\delta{a}"

    def test_vector(self):
        x = Vector("x")
        assert str(Variation(x)) == "\\delta{x}"

    def test_matrix(self):
        M = Matrix("M")
        assert str(Variation(M)) == "\\delta{M}"


class TestVariationProperties:
    def test_is_constant_true(self):
        c = Scalar("c", attr=["Constant"])
        assert Variation(c).is_constant is True

    def test_is_constant_false(self):
        a = Scalar("a")
        assert Variation(a).is_constant is False

    def test_is_zero_true(self):
        z = Vector("0", attr=["Constant", "Zero"])
        assert Variation(z).is_zero is True

    def test_is_zero_false(self):
        x = Vector("x")
        assert Variation(x).is_zero is False


class TestVariationHas:
    def test_has_inner(self):
        x = Vector("x")
        assert Variation(x).has(x) is True

    def test_has_missing(self):
        x = Vector("x")
        y = Vector("y")
        assert Variation(x).has(y) is False


class TestVariationEqHash:
    def test_eq(self):
        x = Vector("x")
        assert Variation(x) == Variation(x)

    def test_neq(self):
        x = Vector("x")
        y = Vector("y")
        assert Variation(x) != Variation(y)

    def test_hash(self):
        x = Vector("x")
        assert hash(Variation(x)) == hash(Variation(x))

    def test_set(self):
        x = Vector("x")
        assert len({Variation(x), Variation(x)}) == 1


class TestVariationOperators:
    def test_add_scalar(self):
        a, b = getScalars("a b")
        result = Variation(a) + b
        assert isinstance(result, Add)

    def test_add_vector(self):
        x, y = getVectors(["x", "y"])
        result = Variation(x) + y
        assert isinstance(result, VAdd)

    def test_add_matrix(self):
        M, N = getMatrices("M N")
        result = Variation(M) + N
        assert isinstance(result, MAdd)

    def test_sub_scalar(self):
        a, b = getScalars("a b")
        result = Variation(a) - b
        assert isinstance(result, Add)

    def test_sub_vector(self):
        x, y = getVectors(["x", "y"])
        result = Variation(x) - y
        assert isinstance(result, VAdd)

    def test_mul_scalar_scalar(self):
        a, b = getScalars("a b")
        result = Variation(a) * b
        assert isinstance(result, Mul)

    def test_mul_vector_scalar(self):
        x = Vector("x")
        a = Scalar("a")
        result = Variation(x) * a
        assert isinstance(result, SVMul)

    def test_mul_numeric(self):
        a = Scalar("a")
        result = Variation(a) * 3
        assert isinstance(result, Mul)


# ===================================================================
# Dot
# ===================================================================


class TestDotType:
    def test_type(self):
        x, y = getVectors(["x", "y"])
        assert Dot(x, y).type == ExprType.SCALAR


class TestDotConstruction:
    def test_left_right(self):
        x, y = getVectors(["x", "y"])
        d = Dot(x, y)
        assert d.left == x
        assert d.right == y

    def test_arity(self):
        x, y = getVectors(["x", "y"])
        assert Dot(x, y).arity == 2

    def test_len(self):
        x, y = getVectors(["x", "y"])
        assert len(Dot(x, y)) == 2


class TestDotStr:
    def test_str(self):
        x, y = getVectors(["x", "y"])
        assert str(Dot(x, y)) == "Dot(x,y)"


class TestDotProperties:
    def test_is_constant_both(self):
        e1 = Vector("e1", attr=["Constant"])
        e2 = Vector("e2", attr=["Constant"])
        assert Dot(e1, e2).is_constant is True

    def test_is_constant_one(self):
        e1 = Vector("e1", attr=["Constant"])
        x = Vector("x")
        assert Dot(e1, x).is_constant is False

    def test_is_zero_left(self):
        z = Vector("0", attr=["Constant", "Zero"])
        x = Vector("x")
        assert Dot(z, x).is_zero is True

    def test_is_zero_right(self):
        x = Vector("x")
        z = Vector("0", attr=["Constant", "Zero"])
        assert Dot(x, z).is_zero is True

    def test_is_zero_neither(self):
        x, y = getVectors(["x", "y"])
        assert Dot(x, y).is_zero is False


class TestDotHas:
    def test_has_left(self):
        x, y = getVectors(["x", "y"])
        assert Dot(x, y).has(x) is True

    def test_has_right(self):
        x, y = getVectors(["x", "y"])
        assert Dot(x, y).has(y) is True

    def test_has_missing(self):
        x, y = getVectors(["x", "y"])
        z = Vector("z")
        assert Dot(x, y).has(z) is False


class TestDotEqHash:
    def test_eq(self):
        x, y = getVectors(["x", "y"])
        assert Dot(x, y) == Dot(x, y)

    def test_neq(self):
        x, y, z = getVectors(["x", "y", "z"])
        assert Dot(x, y) != Dot(x, z)

    def test_hash(self):
        x, y = getVectors(["x", "y"])
        assert hash(Dot(x, y)) == hash(Dot(x, y))


class TestDotTypeMismatch:
    def test_scalar_vector(self):
        a = Scalar("a")
        x = Vector("x")
        with pytest.raises(ExpressionMismatchError):
            Dot(a, x)

    def test_vector_matrix(self):
        x = Vector("x")
        M = Matrix("M")
        with pytest.raises(ExpressionMismatchError):
            Dot(x, M)


# ===================================================================
# Cross
# ===================================================================


class TestCrossType:
    def test_type(self):
        x, y = getVectors(["x", "y"])
        assert Cross(x, y).type == ExprType.VECTOR


class TestCrossConstruction:
    def test_left_right(self):
        x, y = getVectors(["x", "y"])
        c = Cross(x, y)
        assert c.left == x
        assert c.right == y

    def test_arity(self):
        x, y = getVectors(["x", "y"])
        assert Cross(x, y).arity == 2

    def test_len(self):
        x, y = getVectors(["x", "y"])
        assert len(Cross(x, y)) == 2


class TestCrossStr:
    def test_str(self):
        x, y = getVectors(["x", "y"])
        assert str(Cross(x, y)) == "Cross(x,y)"


class TestCrossProperties:
    def test_is_constant_both(self):
        e1 = Vector("e1", attr=["Constant"])
        e2 = Vector("e2", attr=["Constant"])
        assert Cross(e1, e2).is_constant is True

    def test_is_constant_one(self):
        e1 = Vector("e1", attr=["Constant"])
        x = Vector("x")
        assert Cross(e1, x).is_constant is False

    def test_is_zero_left(self):
        z = Vector("0", attr=["Constant", "Zero"])
        x = Vector("x")
        assert Cross(z, x).is_zero is True

    def test_is_zero_right(self):
        x = Vector("x")
        z = Vector("0", attr=["Constant", "Zero"])
        assert Cross(x, z).is_zero is True

    def test_is_zero_neither(self):
        x, y = getVectors(["x", "y"])
        assert Cross(x, y).is_zero is False


class TestCrossHas:
    def test_has_left(self):
        x, y = getVectors(["x", "y"])
        assert Cross(x, y).has(x) is True

    def test_has_missing(self):
        x, y = getVectors(["x", "y"])
        z = Vector("z")
        assert Cross(x, y).has(z) is False


class TestCrossEqHash:
    def test_eq(self):
        x, y = getVectors(["x", "y"])
        assert Cross(x, y) == Cross(x, y)

    def test_neq(self):
        x, y = getVectors(["x", "y"])
        assert Cross(x, y) != Cross(y, x)


class TestCrossTypeMismatch:
    def test_scalar_vector(self):
        a = Scalar("a")
        x = Vector("x")
        with pytest.raises(ExpressionMismatchError):
            Cross(a, x)


# ===================================================================
# Hat
# ===================================================================


class TestHatType:
    def test_type(self):
        x = Vector("x")
        assert Hat(x).type == ExprType.MATRIX


class TestHatConstruction:
    def test_expr(self):
        x = Vector("x")
        assert Hat(x).expr == x

    def test_arity(self):
        x = Vector("x")
        assert Hat(x).arity == 1

    def test_len(self):
        x = Vector("x")
        assert len(Hat(x)) == 1


class TestHatStr:
    def test_str(self):
        x = Vector("x")
        assert str(Hat(x)) == "Hat(x)"


class TestHatProperties:
    def test_is_constant_true(self):
        e1 = Vector("e1", attr=["Constant"])
        assert Hat(e1).is_constant is True

    def test_is_constant_false(self):
        x = Vector("x")
        assert Hat(x).is_constant is False

    def test_is_zero_true(self):
        z = Vector("0", attr=["Constant", "Zero"])
        assert Hat(z).is_zero is True

    def test_is_zero_false(self):
        x = Vector("x")
        assert Hat(x).is_zero is False


class TestHatHas:
    def test_has_inner(self):
        x = Vector("x")
        assert Hat(x).has(x) is True

    def test_has_missing(self):
        x = Vector("x")
        y = Vector("y")
        assert Hat(x).has(y) is False


class TestHatTypeMismatch:
    def test_rejects_scalar(self):
        a = Scalar("a")
        with pytest.raises(ExpressionMismatchError):
            Hat(a)

    def test_rejects_matrix(self):
        M = Matrix("M")
        with pytest.raises(ExpressionMismatchError):
            Hat(M)


# ===================================================================
# Vee
# ===================================================================


class TestVeeType:
    def test_type(self):
        M = Matrix("M")
        assert Vee(M).type == ExprType.VECTOR


class TestVeeConstruction:
    def test_expr(self):
        M = Matrix("M")
        assert Vee(M).expr == M

    def test_arity(self):
        M = Matrix("M")
        assert Vee(M).arity == 1

    def test_len(self):
        M = Matrix("M")
        assert len(Vee(M)) == 1


class TestVeeStr:
    def test_str(self):
        M = Matrix("M")
        assert str(Vee(M)) == "Vee(M)"


class TestVeeProperties:
    def test_is_constant_true(self):
        C = Matrix("C", attr=["Constant"])
        assert Vee(C).is_constant is True

    def test_is_constant_false(self):
        M = Matrix("M")
        assert Vee(M).is_constant is False

    def test_is_zero_true(self):
        Z = Matrix("O", attr=["Constant", "Zero"])
        assert Vee(Z).is_zero is True

    def test_is_zero_false(self):
        M = Matrix("M")
        assert Vee(M).is_zero is False


class TestVeeHas:
    def test_has_inner(self):
        M = Matrix("M")
        assert Vee(M).has(M) is True

    def test_has_missing(self):
        M = Matrix("M")
        N = Matrix("N")
        assert Vee(M).has(N) is False


class TestVeeTypeMismatch:
    def test_rejects_scalar(self):
        a = Scalar("a")
        with pytest.raises(ExpressionMismatchError):
            Vee(a)

    def test_rejects_vector(self):
        x = Vector("x")
        with pytest.raises(ExpressionMismatchError):
            Vee(x)


# ===================================================================
# Transpose
# ===================================================================


class TestTransposeType:
    def test_vector(self):
        x = Vector("x")
        assert Transpose(x).type == ExprType.VECTOR

    def test_matrix(self):
        M = Matrix("M")
        assert Transpose(M).type == ExprType.MATRIX

    def test_scalar(self):
        a = Scalar("a")
        assert Transpose(a).type == ExprType.SCALAR

    def test_none(self):
        assert Transpose().type is None


class TestTransposeConstruction:
    def test_expr(self):
        x = Vector("x")
        assert Transpose(x).expr == x

    def test_expr_none(self):
        assert Transpose().expr is None

    def test_arity(self):
        x = Vector("x")
        assert Transpose(x).arity == 1

    def test_len(self):
        x = Vector("x")
        assert len(Transpose(x)) == 1


class TestTransposeStr:
    def test_vector(self):
        x = Vector("x")
        assert str(Transpose(x)) == "(x)'"

    def test_matrix(self):
        M = Matrix("M")
        assert str(Transpose(M)) == "(M)'"


class TestTransposeProperties:
    def test_is_constant_true(self):
        e1 = Vector("e1", attr=["Constant"])
        assert Transpose(e1).is_constant is True

    def test_is_constant_false(self):
        x = Vector("x")
        assert Transpose(x).is_constant is False

    def test_is_constant_none(self):
        assert Transpose().is_constant is False

    def test_is_zero_true(self):
        z = Vector("0", attr=["Constant", "Zero"])
        assert Transpose(z).is_zero is True

    def test_is_zero_false(self):
        x = Vector("x")
        assert Transpose(x).is_zero is False

    def test_is_zero_none(self):
        assert Transpose().is_zero is False

    def test_size_vector(self):
        x = Vector("x", size=(3,))
        assert Transpose(x).size == (3,)

    def test_size_matrix(self):
        M = Matrix("M", size=(3, 4))
        assert Transpose(M).size == (3, 4)

    def test_size_none(self):
        assert Transpose().size is None


class TestTransposeHas:
    def test_has_inner(self):
        x = Vector("x")
        assert Transpose(x).has(x) is True

    def test_has_missing(self):
        x = Vector("x")
        y = Vector("y")
        assert Transpose(x).has(y) is False


class TestTransposeEqHash:
    def test_eq(self):
        x = Vector("x")
        assert Transpose(x) == Transpose(x)

    def test_neq(self):
        x = Vector("x")
        y = Vector("y")
        assert Transpose(x) != Transpose(y)

    def test_hash(self):
        x = Vector("x")
        assert hash(Transpose(x)) == hash(Transpose(x))


class TestTransposeOperators:
    def test_add_vector(self):
        x, y = getVectors(["x", "y"])
        result = Transpose(x) + y
        assert isinstance(result, VAdd)

    def test_add_matrix(self):
        M, N = getMatrices("M N")
        result = Transpose(M) + N
        assert isinstance(result, MAdd)

    def test_sub_vector(self):
        x, y = getVectors(["x", "y"])
        result = Transpose(x) - y
        assert isinstance(result, VAdd)

    def test_mul_scalar(self):
        x = Vector("x")
        a = Scalar("a")
        result = Transpose(x) * a
        assert isinstance(result, Transpose)
        assert isinstance(result.expr, SVMul)

    def test_mul_vector(self):
        x, y = getVectors(["x", "y"])
        result = Transpose(x) * y
        assert isinstance(result, VVMul)
        assert result.type == ExprType.SCALAR

    def test_mul_matrix(self):
        x = Vector("x")
        M = Matrix("M")
        result = Transpose(x) * M
        assert isinstance(result, MVMul)

    def test_mul_numeric(self):
        x = Vector("x")
        result = Transpose(x) * 2
        assert isinstance(result, Transpose)
