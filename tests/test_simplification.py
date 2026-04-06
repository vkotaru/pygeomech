from geomech.core.base import Scalar, Vector, getVectors
from geomech.core.operations import Add, Cross, Dot, Mul, SVMul, VAdd
from geomech.core.transformations import pull, simplify, vector_rules


class TestSimplifyScalar:
    def test_remove_zeros_from_add(self):
        a = Scalar("a")
        z = Scalar("0", value=0, attr=["Constant", "Zero"])
        expr = Add(a, z)
        result = simplify(expr)
        assert result == a

    def test_mul_with_zero(self):
        a = Scalar("a")
        z = Scalar("0", value=0, attr=["Constant", "Zero"])
        expr = Mul(a, z)
        result = simplify(expr)
        assert result.is_zero

    def test_mul_with_one_left(self):
        a = Scalar("a")
        one = Scalar("1", value=1, attr=["Constant", "Ones"])
        expr = Mul(one, a)
        result = simplify(expr)
        assert result == a

    def test_mul_with_one_right(self):
        a = Scalar("a")
        one = Scalar("1", value=1, attr=["Constant", "Ones"])
        expr = Mul(a, one)
        result = simplify(expr)
        assert result == a


class TestSimplifyVector:
    def test_remove_zeros_from_vadd(self):
        x = Vector("x")
        z = Vector("0", attr=["Constant", "Zero"])
        expr = VAdd(x, z)
        result = simplify(expr)
        assert result == x


class TestPull:
    def test_pull_scalar_from_dot_left(self):
        a = Scalar("a")
        x, y = getVectors(["x", "y"])
        expr = Dot(SVMul(x, a), y)
        result = pull(expr)
        assert isinstance(result, Mul)

    def test_pull_scalar_from_dot_right(self):
        a = Scalar("a")
        x, y = getVectors(["x", "y"])
        expr = Dot(x, SVMul(y, a))
        result = pull(expr)
        assert isinstance(result, Mul)

    def test_pull_scalar_from_cross_left(self):
        a = Scalar("a")
        x, y = getVectors(["x", "y"])
        expr = Cross(SVMul(x, a), y)
        result = pull(expr)
        assert isinstance(result, SVMul)

    def test_pull_scalar_from_cross_right(self):
        a = Scalar("a")
        x, y = getVectors(["x", "y"])
        expr = Cross(x, SVMul(y, a))
        result = pull(expr)
        assert isinstance(result, SVMul)


class TestVectorRules:
    def test_dot_self_unit_norm(self):
        q = Vector("q", attr=["Constant", "UnitNorm"])
        result = vector_rules(Dot(q, q))
        assert result.value == 1

    def test_dot_cross_orthogonality(self):
        """Dot(x, Cross(x, y)) == 0"""
        x, y = getVectors(["x", "y"])
        expr = Dot(x, Cross(x, y))
        result = vector_rules(expr)
        assert result.is_zero
