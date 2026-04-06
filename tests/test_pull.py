from geomech.core.base.expressions import (
    Matrix,
    Scalar,
    Vector,
    getMatrices,
    getVectors,
)
from geomech.core.operations.addition import Add, MAdd, VAdd
from geomech.core.operations.geometry import Cross, Dot
from geomech.core.operations.multiplication import MMMul, Mul, MVMul, SMMul, SVMul
from geomech.core.transformations.pull import pull

# ---------------------------------------------------------------------------
# Scalar: Dot
# ---------------------------------------------------------------------------


class TestPullDot:
    def test_dot_no_scalars(self):
        """dot(x, y) has nothing to pull — returned unchanged."""
        x, y = getVectors(["x", "y"])
        result = pull(Dot(x, y))
        assert isinstance(result, Dot)

    def test_dot_scalar_left(self):
        """dot(a*x, y) → a * dot(x, y)"""
        a = Scalar("a")
        x, y = getVectors(["x", "y"])
        result = pull(Dot(SVMul(x, a), y))
        assert isinstance(result, Mul)
        assert str(a) in str(result)

    def test_dot_scalar_right(self):
        """dot(x, a*y) → a * dot(x, y)"""
        a = Scalar("a")
        x, y = getVectors(["x", "y"])
        result = pull(Dot(x, SVMul(y, a)))
        assert isinstance(result, Mul)
        assert str(result) == "aDot(x,y)"

    def test_dot_scalar_both(self):
        """dot(a*x, b*y) → a * b * dot(x, y)"""
        a, b = Scalar("a"), Scalar("b")
        x, y = getVectors(["x", "y"])
        result = pull(Dot(SVMul(x, a), SVMul(y, b)))
        assert isinstance(result, Mul)
        assert str(result) == "abDot(x,y)"


# ---------------------------------------------------------------------------
# Scalar: Add, Mul
# ---------------------------------------------------------------------------


class TestPullScalar:
    def test_add_passthrough(self):
        """a + b has no nested scalars to pull — recurse into terms."""
        a, b = Scalar("a"), Scalar("b")
        result = pull(Add(a, b))
        assert isinstance(result, Add)

    def test_mul_passthrough(self):
        """a * b is already scalar — recurse into operands."""
        a, b = Scalar("a"), Scalar("b")
        result = pull(Mul(a, b))
        assert isinstance(result, Mul)

    def test_leaf_passthrough(self):
        """Leaf scalar returned as-is."""
        a = Scalar("a")
        assert pull(a) is a


# ---------------------------------------------------------------------------
# Vector: Cross
# ---------------------------------------------------------------------------


class TestPullCross:
    def test_cross_no_scalars(self):
        """cross(x, y) has nothing to pull — returned unchanged."""
        x, y = getVectors(["x", "y"])
        result = pull(Cross(x, y))
        assert isinstance(result, Cross)

    def test_cross_scalar_left(self):
        """cross(a*x, y) → a * cross(x, y)"""
        a = Scalar("a")
        x, y = getVectors(["x", "y"])
        result = pull(Cross(SVMul(x, a), y))
        assert isinstance(result, SVMul)
        assert isinstance(result.left, Cross)
        assert result.right == a
        assert str(result) == "Cross(x,y)a"

    def test_cross_scalar_right(self):
        """cross(x, a*y) → a * cross(x, y)"""
        a = Scalar("a")
        x, y = getVectors(["x", "y"])
        result = pull(Cross(x, SVMul(y, a)))
        assert isinstance(result, SVMul)
        assert isinstance(result.left, Cross)
        assert result.right == a
        assert str(result) == "Cross(x,y)a"

    def test_cross_scalar_both(self):
        """cross(a*x, b*y) → (a*b) * cross(x, y)"""
        a, b = Scalar("a"), Scalar("b")
        x, y = getVectors(["x", "y"])
        result = pull(Cross(SVMul(x, a), SVMul(y, b)))
        assert isinstance(result, SVMul)
        assert isinstance(result.left, Cross)
        assert isinstance(result.right, Mul)
        assert str(result) == "Cross(x,y)ab"


# ---------------------------------------------------------------------------
# Vector: SVMul
# ---------------------------------------------------------------------------


class TestPullSVMul:
    def test_svmul_flat(self):
        """a*x is already pulled — returned unchanged."""
        a = Scalar("a")
        x = Vector("x")
        result = pull(SVMul(x, a))
        assert isinstance(result, SVMul)

    def test_svmul_nested(self):
        """(a*x)*b → x*(a*b) — flatten nested scalar-vector muls."""
        a, b = Scalar("a"), Scalar("b")
        x = Vector("x")
        result = pull(SVMul(SVMul(x, a), b))
        assert isinstance(result, SVMul)
        assert isinstance(result.right, Mul)
        assert str(result.left) == "x"

    def test_svmul_deep_nested(self):
        """((a*x)*b)*c → x*(a*b*c) — three levels of nesting."""
        a, b, c = Scalar("a"), Scalar("b"), Scalar("c")
        x = Vector("x")
        result = pull(SVMul(SVMul(SVMul(x, a), b), c))
        assert isinstance(result, SVMul)
        assert str(result.left) == "x"


# ---------------------------------------------------------------------------
# Vector: MVMul
# ---------------------------------------------------------------------------


class TestPullMVMul:
    def test_mvmul_no_scalars(self):
        """M*x has nothing to pull — returned unchanged."""
        M = Matrix("M")
        x = Vector("x")
        result = pull(MVMul(M, x))
        assert isinstance(result, MVMul)

    def test_mvmul_scalar_left(self):
        """(a*M)*x → (M*x)*a — pull scalar from matrix side."""
        a = Scalar("a")
        M = Matrix("M")
        x = Vector("x")
        result = pull(MVMul(SMMul(M, a), x))
        assert isinstance(result, SVMul)
        assert isinstance(result.left, MVMul)
        assert str(result) == "Mxa"

    def test_mvmul_scalar_right(self):
        """M*(a*x) → (M*x)*a — pull scalar from vector side."""
        a = Scalar("a")
        M = Matrix("M")
        x = Vector("x")
        result = pull(MVMul(M, SVMul(x, a)))
        assert isinstance(result, SVMul)
        assert isinstance(result.left, MVMul)
        assert str(result) == "Mxa"

    def test_mvmul_scalar_both(self):
        """(a*M)*(b*x) → (M*x)*(a*b) — pull scalars from both sides."""
        a, b = Scalar("a"), Scalar("b")
        M = Matrix("M")
        x = Vector("x")
        result = pull(MVMul(SMMul(M, a), SVMul(x, b)))
        assert isinstance(result, SVMul)
        assert isinstance(result.left, MVMul)
        assert isinstance(result.right, Mul)
        assert str(result) == "Mxab"


# ---------------------------------------------------------------------------
# Vector: VAdd
# ---------------------------------------------------------------------------


class TestPullVAdd:
    def test_vadd_passthrough(self):
        """x + y has nothing to pull — recurse into terms."""
        x, y = getVectors(["x", "y"])
        result = pull(VAdd(x, y))
        assert isinstance(result, VAdd)

    def test_vadd_pulls_terms(self):
        """((a*x)*a) + y → (x*a^2) + y — pull applied to each term."""
        a = Scalar("a")
        x, y = getVectors(["x", "y"])
        result = pull(VAdd(SVMul(SVMul(x, a), a), y))
        assert isinstance(result, VAdd)
        assert str(result) == "(xaa+y)"


# ---------------------------------------------------------------------------
# Matrix: SMMul
# ---------------------------------------------------------------------------


class TestPullSMMul:
    def test_smmul_flat(self):
        """a*M is already pulled — returned unchanged."""
        a = Scalar("a")
        M = Matrix("M")
        result = pull(SMMul(M, a))
        assert isinstance(result, SMMul)

    def test_smmul_nested(self):
        """(a*M)*b → M*(a*b) — flatten nested scalar-matrix muls."""
        a, b = Scalar("a"), Scalar("b")
        M = Matrix("M")
        result = pull(SMMul(SMMul(M, a), b))
        assert isinstance(result, SMMul)
        assert isinstance(result.right, Mul)
        assert str(result.left) == "M"
        assert str(result) == "Mab"


# ---------------------------------------------------------------------------
# Matrix: MMMul
# ---------------------------------------------------------------------------


class TestPullMMMul:
    def test_mmmul_no_scalars(self):
        """M*N has nothing to pull — returned unchanged."""
        M, N = getMatrices(["M", "N"])
        result = pull(MMMul(M, N))
        assert isinstance(result, MMMul)

    def test_mmmul_scalar_left(self):
        """(a*M)*N → (M*N)*a — pull scalar from left matrix."""
        a = Scalar("a")
        M, N = getMatrices(["M", "N"])
        result = pull(MMMul(SMMul(M, a), N))
        assert isinstance(result, SMMul)
        assert isinstance(result.left, MMMul)
        assert str(result) == "MNa"

    def test_mmmul_scalar_right(self):
        """M*(a*N) → (M*N)*a — pull scalar from right matrix."""
        a = Scalar("a")
        M, N = getMatrices(["M", "N"])
        result = pull(MMMul(M, SMMul(N, a)))
        assert isinstance(result, SMMul)
        assert isinstance(result.left, MMMul)
        assert str(result) == "MNa"

    def test_mmmul_scalar_both(self):
        """(a*M)*(b*N) → (M*N)*(a*b) — pull scalars from both sides."""
        a, b = Scalar("a"), Scalar("b")
        M, N = getMatrices(["M", "N"])
        result = pull(MMMul(SMMul(M, a), SMMul(N, b)))
        assert isinstance(result, SMMul)
        assert isinstance(result.left, MMMul)
        assert isinstance(result.right, Mul)
        assert str(result) == "MNab"


# ---------------------------------------------------------------------------
# Matrix: MAdd
# ---------------------------------------------------------------------------


class TestPullMAdd:
    def test_madd_passthrough(self):
        """M + N has nothing to pull — recurse into terms."""
        M, N = getMatrices(["M", "N"])
        result = pull(MAdd(M, N))
        assert isinstance(result, MAdd)


# ---------------------------------------------------------------------------
# Deep nesting (bottom-up correctness)
# ---------------------------------------------------------------------------


class TestPullDeepNesting:
    def test_dot_with_cross_containing_scalar(self):
        """dot(cross(a*x, y), z) → bottom-up: first pull cross, then dot.
        cross(a*x, y) → a*cross(x,y), then dot(a*cross(x,y), z) → a*dot(cross(x,y), z)"""
        a = Scalar("a")
        x, y, z = getVectors(["x", "y", "z"])
        expr = Dot(Cross(SVMul(x, a), y), z)
        result = pull(expr)
        assert isinstance(result, Mul)
        assert str(result) == "aDot(Cross(x,y),z)"

    def test_mvmul_with_nested_smmul(self):
        """(a*M)*(b*x) → (M*x)*(a*b) — scalars from both sides combined."""
        a, b = Scalar("a"), Scalar("b")
        M = Matrix("M")
        x = Vector("x")
        expr = MVMul(SMMul(M, a), SVMul(x, b))
        result = pull(expr)
        assert isinstance(result, SVMul)
        assert isinstance(result.left, MVMul)
        assert str(result.left.left) == "M"
        assert str(result.left.right) == "x"
        assert str(result) == "Mxab"

    def test_leaf_vector_passthrough(self):
        """Leaf vector returned as-is (identity case)."""
        x = Vector("x")
        assert pull(x) is x

    def test_leaf_matrix_passthrough(self):
        """Leaf matrix returned as-is (identity case)."""
        M = Matrix("M")
        assert pull(M) is M
