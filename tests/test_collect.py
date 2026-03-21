"""Tests for geomech.core.math.collect — rearrange Dot so target vec is on the left."""

import pytest

from geomech.core.base.expressions import Scalar, Vector, Matrix
from geomech.core.operations.addition import Add
from geomech.core.operations.multiplication import Mul, MVMul
from geomech.core.operations.geometry import Dot, Cross
from geomech.core.math.collect import collect


@pytest.fixture
def vectors():
    v = Vector('v')
    w = Vector('w')
    u = Vector('u')
    return v, w, u


@pytest.fixture
def matrices():
    M = Matrix('M')
    return M


# ===========================================================================
# Passthrough cases
# ===========================================================================

class TestPassthrough:
    """Expressions that don't contain vec or aren't Dot return unchanged."""

    def test_scalar_leaf(self, vectors):
        """Scalar leaf passes through."""
        v, _, _ = vectors
        s = Scalar('s')
        assert collect(s, v) is s

    def test_dot_neither_has_vec(self, vectors):
        """Dot(w, u) where neither is vec returns unchanged."""
        v, w, u = vectors
        expr = Dot(w, u)
        result = collect(expr, v)
        assert str(result) == str(expr)


# ===========================================================================
# Simple flip / already canonical
# ===========================================================================

class TestSimpleFlip:
    """Dot(_, vec) flips to Dot(vec, _)."""

    def test_vec_on_left_unchanged(self, vectors):
        """Dot(vec, w) is already canonical."""
        v, w, _ = vectors
        expr = Dot(v, w)
        result = collect(expr, v)
        assert str(result.left) == 'v'
        assert str(result.right) == 'w'

    def test_vec_on_right_flips(self, vectors):
        """Dot(w, vec) flips to Dot(vec, w)."""
        v, w, _ = vectors
        expr = Dot(w, v)
        result = collect(expr, v)
        assert str(result.left) == 'v'
        assert str(result.right) == 'w'


# ===========================================================================
# Add / Mul recursion
# ===========================================================================

class TestRecursion:
    """collect recurses through Add and Mul."""

    def test_add_collects_each_term(self, vectors):
        """Add(Dot(w, vec), Dot(vec, u)) → both terms have vec on left."""
        v, w, u = vectors
        expr = Add(Dot(w, v), Dot(v, u))
        result = collect(expr, v)
        assert isinstance(result, Add)
        # First term should be flipped, second already canonical
        assert str(result.nodes[0].left) == 'v'
        assert str(result.nodes[1].left) == 'v'

    def test_mul_collects_each_side(self, vectors):
        """Mul(s, Dot(w, vec)) → Dot is collected inside Mul."""
        v, w, _ = vectors
        s = Scalar('s')
        expr = Mul(s, Dot(w, v))
        result = collect(expr, v)
        assert isinstance(result, Mul)
        # Right side should have vec on left of Dot
        assert isinstance(result.right, Dot)
        assert str(result.right.left) == 'v'


# ===========================================================================
# Cross — scalar triple product
# ===========================================================================

class TestCrossPatterns:
    """Dot(_, Cross(..)) rearranged via scalar triple product."""

    def test_cross_right_left_is_vec(self, vectors):
        """Dot(w, Cross(vec, u)) → Dot(vec, Cross(u, w))."""
        v, w, u = vectors
        expr = Dot(w, Cross(v, u))
        result = collect(expr, v)
        assert str(result.left) == 'v'
        assert isinstance(result.right, Cross)
        assert str(result.right.left) == 'u'
        assert str(result.right.right) == 'w'

    def test_cross_right_right_is_vec(self, vectors):
        """Dot(w, Cross(u, vec)) → Dot(vec, Cross(w, u))."""
        v, w, u = vectors
        expr = Dot(w, Cross(u, v))
        result = collect(expr, v)
        assert str(result.left) == 'v'
        assert isinstance(result.right, Cross)
        assert str(result.right.left) == 'w'
        assert str(result.right.right) == 'u'

    def test_cross_neither_has_vec(self, vectors):
        """Dot(w, Cross(u, w)) where vec not present returns unchanged."""
        v, w, u = vectors
        expr = Dot(u, Cross(w, u))
        result = collect(expr, v)
        assert str(result) == str(expr)


# ===========================================================================
# MVMul patterns
# ===========================================================================

class TestMVMulPatterns:
    """Dot with MVMul rearranged to put vec on left."""

    def test_mvmul_left_vec_in_right(self, vectors, matrices):
        """Dot(M*vec, w) → Dot(vec, M^T*w)."""
        v, w, _ = vectors
        M = matrices
        expr = Dot(MVMul(M, v), w)
        result = collect(expr, v)
        assert str(result.left) == 'v'
        assert isinstance(result.right, MVMul)
        assert str(result.right.right) == 'w'

    def test_mvmul_right_flips(self, vectors, matrices):
        """Dot(w, M*vec) → flips to Dot(M*vec, w) then → Dot(vec, M*w)."""
        v, w, _ = vectors
        M = matrices
        expr = Dot(w, MVMul(M, v))
        result = collect(expr, v)
        assert str(result.left) == 'v'
        assert isinstance(result.right, MVMul)


# ===========================================================================
# MVMul + Cross (kinetic energy patterns)
# ===========================================================================

class TestKineticEnergyPatterns:
    """Dot(MVMul, Cross) patterns from kinetic energy terms."""

    def test_cross_mvmul_flips(self, vectors, matrices):
        """Dot(Cross(..), M*vec) flips to put MVMul on left, then rearranges."""
        v, w, u = vectors
        M = matrices
        expr = Dot(Cross(w, u), MVMul(M, v))
        result = collect(expr, v)
        # Flips to Dot(MVMul(M,v), Cross(w,u)), then MVMul.right == v
        # → Dot(v, Cross(MVMul(M, w), u))
        assert str(result.left) == 'v'

    def test_mvmul_cross_vec_in_cross_left(self, vectors, matrices):
        """Dot(M*a, Cross(vec, c)) → Dot(vec, Cross(c, M*a))."""
        v, w, u = vectors
        M = matrices
        expr = Dot(MVMul(M, w), Cross(v, u))
        result = collect(expr, v)
        assert str(result.left) == 'v'
        assert isinstance(result.right, Cross)
        assert str(result.right.left) == 'u'

    def test_mvmul_cross_vec_in_cross_right(self, vectors, matrices):
        """Dot(M*a, Cross(b, vec)) → Dot(vec, Cross(M*a, b))."""
        v, w, u = vectors
        M = matrices
        expr = Dot(MVMul(M, w), Cross(u, v))
        result = collect(expr, v)
        assert str(result.left) == 'v'
        assert isinstance(result.right, Cross)

    def test_mvmul_cross_vec_in_mvmul(self, vectors, matrices):
        """Dot(M*vec, Cross(b, c)) → Dot(vec, Cross(M*b, c))."""
        v, w, u = vectors
        M = matrices
        expr = Dot(MVMul(M, v), Cross(w, u))
        result = collect(expr, v)
        assert str(result.left) == 'v'
        assert isinstance(result.right, Cross)
        # Cross(MVMul(M, w), u)
        assert isinstance(result.right.left, MVMul)
        assert str(result.right.right) == 'u'
