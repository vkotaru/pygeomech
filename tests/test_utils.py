from geomech.core.base.expressions import Scalar, Vector, Matrix, getVectors
from geomech.core.operations.addition import Add, VAdd, MAdd
from geomech.core.operations.multiplication import Mul, SVMul, MVMul, MMMul
from geomech.core.operations.geometry import Dot, Cross
from geomech.core.transformations.utils import is_leaf, has_nested_add


class TestIsLeaf:
    def test_scalar_is_leaf(self):
        """Scalar('a') is a leaf — no children."""
        assert is_leaf(Scalar('a'))

    def test_vector_is_leaf(self):
        """Vector('x') is a leaf — no children."""
        assert is_leaf(Vector('x'))

    def test_matrix_is_leaf(self):
        """Matrix('M') is a leaf — no children."""
        assert is_leaf(Matrix('M'))

    def test_add_is_not_leaf(self):
        """Add(a, b) is n-ary — not a leaf."""
        a, b = Scalar('a'), Scalar('b')
        assert not is_leaf(Add(a, b))

    def test_mul_is_not_leaf(self):
        """Mul(a, b) is binary — not a leaf."""
        a, b = Scalar('a'), Scalar('b')
        assert not is_leaf(Mul(a, b))

    def test_dot_is_not_leaf(self):
        """Dot(x, y) is binary — not a leaf."""
        x, y = getVectors(['x', 'y'])
        assert not is_leaf(Dot(x, y))


class TestHasNestedAdd:
    def test_add_returns_true(self):
        """Top-level Add is itself an addition."""
        a, b = Scalar('a'), Scalar('b')
        assert has_nested_add(Add(a, b))

    def test_vadd_returns_true(self):
        """Top-level VAdd is itself an addition."""
        x, y = getVectors(['x', 'y'])
        assert has_nested_add(VAdd(x, y))

    def test_leaf_returns_false(self):
        """Leaf scalar has no nested structure."""
        assert not has_nested_add(Scalar('a'))

    def test_mul_without_add_returns_false(self):
        """Mul(a, b) with leaf operands — no addition nested."""
        a, b = Scalar('a'), Scalar('b')
        assert not has_nested_add(Mul(a, b))

    def test_mul_with_nested_add(self):
        """(a + b) * c — Add is nested inside Mul."""
        a, b, c = Scalar('a'), Scalar('b'), Scalar('c')
        expr = Mul(Add(a, b), c)
        assert has_nested_add(expr)

    def test_dot_with_nested_vadd(self):
        """dot(x + y, z) — VAdd is nested inside Dot."""
        x, y, z = getVectors(['x', 'y', 'z'])
        expr = Dot(VAdd(x, y), z)
        assert has_nested_add(expr)

    def test_cross_without_add_returns_false(self):
        """cross(x, y) with leaf operands — no addition nested."""
        x, y = getVectors(['x', 'y'])
        assert not has_nested_add(Cross(x, y))
