from geomech.core.base.expressions import (
    Scalar, Vector, ZeroVector, ZeroMatrix, getVectors,
)
from geomech.core.operations.addition import Add
from geomech.core.operations.multiplication import Mul
from geomech.core.operations.geometry import Dot, Cross, Hat, Transpose
from geomech.core.transformations.vector_rules import vector_rules


# ---------------------------------------------------------------------------
# Self-cross: cross(x, x) = 0
# ---------------------------------------------------------------------------

class TestSelfCross:
    def test_cross_self_is_zero(self):
        """cross(x, x) = 0 — cross product of a vector with itself."""
        x = Vector('x')
        result = vector_rules(Cross(x, x))
        assert result is ZeroVector

    def test_cross_different_unchanged(self):
        """cross(x, y) unchanged when x != y."""
        x, y = getVectors(['x', 'y'])
        result = vector_rules(Cross(x, y))
        assert isinstance(result, Cross)


# ---------------------------------------------------------------------------
# Zero absorption: cross(x, 0) = 0, cross(0, x) = 0
# ---------------------------------------------------------------------------

class TestCrossZeroAbsorption:
    def test_cross_zero_right(self):
        """cross(x, 0) = 0 — cross with zero vector."""
        x = Vector('x')
        result = vector_rules(Cross(x, ZeroVector))
        assert result is ZeroVector

    def test_cross_zero_left(self):
        """cross(0, x) = 0 — cross with zero vector."""
        x = Vector('x')
        result = vector_rules(Cross(ZeroVector, x))
        assert result is ZeroVector


# ---------------------------------------------------------------------------
# Unit norm: dot(q, q) = 1
# ---------------------------------------------------------------------------

class TestUnitNorm:
    def test_dot_self_unit_norm(self):
        """dot(q, q) = 1 when q has unit norm."""
        q = Vector('q', attr=['Constant', 'UnitNorm'])
        result = vector_rules(Dot(q, q))
        assert result.value == 1

    def test_dot_self_not_unit_norm(self):
        """dot(x, x) unchanged when x is not unit norm."""
        x = Vector('x')
        result = vector_rules(Dot(x, x))
        assert isinstance(result, Dot)

    def test_dot_different_unit_norm(self):
        """dot(q1, q2) unchanged even if both unit norm but different."""
        q1 = Vector('q1', attr=['UnitNorm'])
        q2 = Vector('q2', attr=['UnitNorm'])
        result = vector_rules(Dot(q1, q2))
        assert isinstance(result, Dot)


# ---------------------------------------------------------------------------
# Zero absorption: dot(x, 0) = 0, dot(0, x) = 0
# ---------------------------------------------------------------------------

class TestDotZeroAbsorption:
    def test_dot_zero_right(self):
        """dot(x, 0) = 0 — dot with zero vector."""
        x = Vector('x')
        result = vector_rules(Dot(x, ZeroVector))
        assert result.isZero

    def test_dot_zero_left(self):
        """dot(0, x) = 0 — dot with zero vector."""
        x = Vector('x')
        result = vector_rules(Dot(ZeroVector, x))
        assert result.isZero


# ---------------------------------------------------------------------------
# Orthogonality: dot(x, cross(x, y)) = 0
# ---------------------------------------------------------------------------

class TestOrthogonality:
    def test_dot_x_cross_x_y(self):
        """dot(x, cross(x, y)) = 0 — x is perpendicular to x x y."""
        x, y = getVectors(['x', 'y'])
        result = vector_rules(Dot(x, Cross(x, y)))
        assert result.isZero

    def test_dot_x_cross_y_x(self):
        """dot(x, cross(y, x)) = 0 — x is perpendicular to y x x."""
        x, y = getVectors(['x', 'y'])
        result = vector_rules(Dot(x, Cross(y, x)))
        assert result.isZero

    def test_dot_x_cross_y_y(self):
        """dot(x, cross(y, y)) = 0 — cross of vector with itself is zero."""
        x, y = getVectors(['x', 'y'])
        result = vector_rules(Dot(x, Cross(y, y)))
        assert result.isZero

    def test_dot_x_cross_y_z_unchanged(self):
        """dot(x, cross(y, z)) unchanged — no orthogonality relation."""
        x, y, z = getVectors(['x', 'y', 'z'])
        result = vector_rules(Dot(x, Cross(y, z)))
        assert isinstance(result, Dot)


# ---------------------------------------------------------------------------
# Cross on left side of dot (commutativity fix)
# ---------------------------------------------------------------------------

class TestOrthogonalityCrossLeft:
    def test_dot_cross_x_y_x(self):
        """dot(cross(x, y), x) = 0 — same rule, cross on left."""
        x, y = getVectors(['x', 'y'])
        result = vector_rules(Dot(Cross(x, y), x))
        assert result.isZero

    def test_dot_cross_y_x_x(self):
        """dot(cross(y, x), x) = 0 — same rule, cross on left."""
        x, y = getVectors(['x', 'y'])
        result = vector_rules(Dot(Cross(y, x), x))
        assert result.isZero

    def test_dot_cross_y_y_x(self):
        """dot(cross(y, y), x) = 0 — self-cross is zero."""
        x, y = getVectors(['x', 'y'])
        result = vector_rules(Dot(Cross(y, y), x))
        assert result.isZero

    def test_dot_cross_y_z_x_unchanged(self):
        """dot(cross(y, z), x) unchanged — no orthogonality relation."""
        x, y, z = getVectors(['x', 'y', 'z'])
        result = vector_rules(Dot(Cross(y, z), x))
        assert isinstance(result, Dot)


# ---------------------------------------------------------------------------
# Double transpose: Transpose(Transpose(x)) = x
# ---------------------------------------------------------------------------

class TestDoubleTranspose:
    def test_double_transpose_cancels(self):
        """(x')' = x — double transpose cancellation."""
        x = Vector('x')
        result = vector_rules(Transpose(Transpose(x)))
        assert result == x
        assert not isinstance(result, Transpose)

    def test_single_transpose_unchanged(self):
        """x' unchanged — single transpose stays."""
        x = Vector('x')
        result = vector_rules(Transpose(x))
        assert isinstance(result, Transpose)


# ---------------------------------------------------------------------------
# Hat of zero: Hat(0) = ZeroMatrix
# ---------------------------------------------------------------------------

class TestHatZero:
    def test_hat_zero_is_zero_matrix(self):
        """Hat(0) = ZeroMatrix — hat map of zero vector."""
        result = vector_rules(Hat(ZeroVector))
        assert result is ZeroMatrix

    def test_hat_nonzero_unchanged(self):
        """Hat(x) unchanged for non-zero vector."""
        x = Vector('x')
        result = vector_rules(Hat(x))
        assert isinstance(result, Hat)


# ---------------------------------------------------------------------------
# Linearity through Add and Mul
# ---------------------------------------------------------------------------

class TestLinearity:
    def test_add_recurses(self):
        """vector_rules applied to each term of a scalar sum."""
        x, y = getVectors(['x', 'y'])
        expr = Add(Dot(x, Cross(x, y)), Scalar('a'))
        result = vector_rules(expr)
        assert isinstance(result, Add)

    def test_mul_recurses(self):
        """vector_rules applied to each operand of a scalar product."""
        x, y = getVectors(['x', 'y'])
        a = Scalar('a')
        expr = Mul(a, Dot(x, Cross(x, y)))
        result = vector_rules(expr)
        assert isinstance(result, Mul)

    def test_leaf_passthrough(self):
        """Leaf expressions returned as-is."""
        a = Scalar('a')
        assert vector_rules(a) is a

    def test_vector_passthrough(self):
        """Non-scalar expressions returned as-is."""
        x = Vector('x')
        assert vector_rules(x) is x
