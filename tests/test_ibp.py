"""Tests for geomech.core.math.ibp — integration by parts."""

import pytest

from geomech.core.base.expressions import Scalar, Vector
from geomech.core.operations.addition import Add
from geomech.core.operations.multiplication import Mul
from geomech.core.operations.geometry import Dot
from geomech.core.math.ibp import _apply_ibp, integrate_by_parts


@pytest.fixture
def vectors():
    v = Vector('v')
    w = Vector('w')
    u = Vector('u')
    return v, w, u


@pytest.fixture
def dot_vectors():
    """Time-derivative vectors and their integrals."""
    dot_eta = Vector('dot_eta')
    eta = Vector('eta')
    return dot_eta, eta


# ===========================================================================
# _apply_ibp — single pass
# ===========================================================================

class TestIbpPassthrough:
    """Expressions that don't match the IBP pattern return unchanged."""

    def test_scalar_leaf(self, vectors):
        """Scalar leaf passes through."""
        v, _, _ = vectors
        s = Scalar('s')
        assert _apply_ibp(s, v) is s

    def test_dot_left_not_target(self, vectors):
        """Dot(w, u) where left != target returns unchanged."""
        v, w, u = vectors
        expr = Dot(w, u)
        result = _apply_ibp(expr, v)
        assert str(result) == str(expr)

    def test_vector_leaf(self, vectors):
        """Vector leaf passes through."""
        v, w, _ = vectors
        assert _apply_ibp(w, v) is w


class TestIbpDot:
    """IBP on Dot where left == target."""

    def test_dot_left_is_target(self, dot_vectors):
        """Dot(dot_eta, w) → Dot(-eta, dot_w)."""
        dot_eta, eta = dot_vectors
        w = Vector('w')
        expr = Dot(dot_eta, w)
        result = _apply_ibp(expr, dot_eta)
        # Result should be Dot(-integrate(dot_eta), diff(w))
        assert isinstance(result, Dot)
        # Left side: dot_eta.t_integrate() * (-1)
        # dot_eta.t_integrate() should give eta
        # Right side: w.t_diff() should give dot_w
        assert result.left.has(eta)
        assert 'dot_w' in str(result.right)


class TestIbpRecursion:
    """IBP recurses through Add and Mul."""

    def test_add_recurses(self, dot_vectors):
        """Add(Dot(target, w), Dot(target, u)) applies IBP to both terms."""
        dot_eta, eta = dot_vectors
        w = Vector('w')
        u = Vector('u')
        expr = Add(Dot(dot_eta, w), Dot(dot_eta, u))
        result = _apply_ibp(expr, dot_eta)
        assert isinstance(result, Add)
        assert len(result.nodes) == 2
        # Both terms should have been transformed
        assert isinstance(result.nodes[0], Dot)
        assert isinstance(result.nodes[1], Dot)

    def test_mul_recurses(self, dot_vectors):
        """Mul(s, Dot(target, w)) applies IBP inside Mul."""
        dot_eta, _ = dot_vectors
        w = Vector('w')
        s = Scalar('s')
        expr = Mul(s, Dot(dot_eta, w))
        result = _apply_ibp(expr, dot_eta)
        assert isinstance(result, Mul)
        # Left side (scalar) unchanged, right side (Dot) transformed
        assert str(result.left) == 's'
        assert isinstance(result.right, Dot)

    def test_add_mixed_terms(self, dot_vectors):
        """Add with some terms matching target and some not."""
        dot_eta, _ = dot_vectors
        w = Vector('w')
        u = Vector('u')
        # Dot(dot_eta, w) + Dot(u, w)  — only first term gets transformed
        expr = Add(Dot(dot_eta, w), Dot(u, w))
        result = _apply_ibp(expr, dot_eta)
        assert isinstance(result, Add)
        # Second term should be unchanged
        assert str(result.nodes[1].left) == 'u'


# ===========================================================================
# integrate_by_parts — full pipeline
# ===========================================================================

class TestIntegrateByParts:
    """Full IBP pipeline: expand → simplify → collect → _apply_ibp → simplify."""

    def test_simple_dot(self, dot_vectors):
        """Simple Dot(dot_eta, w) through the full pipeline.

        IBP gives Dot(-eta, dot_w), simplify pulls -1 out → Mul(-1, Dot(eta, dot_w)).
        """
        dot_eta, eta = dot_vectors
        w = Vector('w')
        expr = Dot(dot_eta, w)
        result = integrate_by_parts(expr, [dot_eta])
        # full_simplify pulls the -1 scalar out of the Dot
        assert isinstance(result, Mul)
        assert result.left.value == -1
        assert isinstance(result.right, Dot)
        assert str(result.right.left) == 'eta'
        assert str(result.right.right) == 'dot_w'

    def test_flipped_dot_gets_collected_then_ibpd(self, dot_vectors):
        """Dot(w, dot_eta) — collect flips it, then IBP applies."""
        dot_eta, eta = dot_vectors
        w = Vector('w')
        expr = Dot(w, dot_eta)
        result = integrate_by_parts(expr, [dot_eta])
        # collect flips to Dot(dot_eta, w), then IBP → Mul(-1, Dot(eta, dot_w))
        assert isinstance(result, Mul)
        assert result.left.value == -1
        assert isinstance(result.right, Dot)
