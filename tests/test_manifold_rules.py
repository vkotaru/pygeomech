"""Tests for manifold kinematic substitutions (manifold_rules.py).

Tests cover:
  - S2 manifold: δ(ω) = ξ̇ - ω × ξ
  - SO3 manifold: δ(Ω) = η̇ + Ω × η
  - Mixed systems with both S2 and SO3
  - Nested variations inside compound expressions
  - Non-manifold expressions pass through unchanged
"""

import pytest
from geomech.core.base.expressions import (
    Scalar,
    Vector,
    Matrix,
    S2,
    SO3,
    TS2,
    TSO3,
    Zero,
    ZeroVector,
)
from geomech.core.operations.addition import VAdd
from geomech.core.operations.multiplication import SVMul, MVMul, MMMul
from geomech.core.operations.geometry import Dot, Cross, Hat
from geomech.core.operations.calculus import Variation, TimeDerivative
from geomech.core.transformations.manifold_rules import apply_manifold_rules
from geomech.dynamics.variables import SystemVariables

# ---------------------------------------------------------------------------
# S2 manifold rules
# ---------------------------------------------------------------------------


class TestS2ManifoldRules:

    def setup_method(self):
        self.q = S2('q')
        self.omega = self.q.get_tangent_vector()
        self.xi = self.q.get_variation_vector()
        self.variables = SystemVariables(vectors=[self.q])

    def test_variation_of_omega_expands(self):
        """δ(ω) → ξ̇ - ω × ξ."""
        expr = Variation(self.omega)
        result = apply_manifold_rules(expr, self.variables)
        assert isinstance(result, VAdd)
        assert len(result.nodes) == 2
        # First term: TimeDerivative(xi)
        assert isinstance(result.nodes[0], TimeDerivative)
        assert str(result.nodes[0].expr) == str(self.xi)
        # Second term: -1 * Cross(omega, xi)
        assert isinstance(result.nodes[1], SVMul)
        assert result.nodes[1].right == Scalar('(-1)',
                                               value=-1,
                                               attr=['Constant'])
        assert result.nodes[1].left.has(self.omega)  # Cross contains omega
        assert result.nodes[1].left.has(self.xi)  # Cross contains xi

    def test_variation_of_xi_raises(self):
        """δ(ξ) is nonsensical — ξ is already a variation vector."""
        with pytest.raises(ValueError, match='variation vector'):
            Variation(self.xi)

    def test_non_manifold_vector_unchanged(self):
        """Variation of a plain vector passes through."""
        x = Vector('x')
        expr = Variation(x)
        result = apply_manifold_rules(expr, self.variables)
        assert isinstance(result, Variation)
        assert str(result.expr) == 'x'

    def test_nested_in_dot(self):
        """δ(ω) inside Dot should be expanded."""
        v = Vector('v', attr=['Constant'])
        expr = Dot(Variation(self.omega), v)
        result = apply_manifold_rules(expr, self.variables)
        # The Dot should be rebuilt with expanded variation
        assert result.has(self.xi)

    def test_nested_in_addition(self):
        """δ(ω) inside VAdd should be expanded."""
        v = Vector('v')
        expr = VAdd(Variation(self.omega), v)
        result = apply_manifold_rules(expr, self.variables)
        assert not isinstance(result.nodes[0], Variation)

    def test_omega_is_ts2(self):
        """Tangent and variation vectors should be TS2 with parent ref."""
        assert isinstance(self.omega, TS2)
        assert isinstance(self.xi, TS2)
        assert self.omega.S2 is self.q
        assert self.xi.S2 is self.q

    def test_s2_delta_is_cross_xi_q(self):
        """q.delta() → ξ × q (variation of manifold point)."""
        result = self.q.delta()
        assert isinstance(result, Cross)
        assert str(result.left) == str(self.xi)
        assert str(result.right) == str(self.q)

    def test_s2_t_diff_is_cross_omega_q(self):
        """q.t_diff() → ω × q (kinematics equation)."""
        result = self.q.t_diff()
        assert isinstance(result, Cross)
        assert str(result.left) == str(self.omega)
        assert str(result.right) == str(self.q)

    def test_no_manifold_vectors_passthrough(self):
        """If no manifold variables, expression passes through unchanged."""
        x = Vector('x')
        variables = SystemVariables(vectors=[x])
        expr = Variation(x)
        result = apply_manifold_rules(expr, variables)
        assert str(result) == str(expr)


# ---------------------------------------------------------------------------
# SO3 manifold rules
# ---------------------------------------------------------------------------


class TestSO3ManifoldRules:

    def setup_method(self):
        self.R = SO3('R')
        self.Omega = self.R.get_tangent_vector()
        self.eta = self.R.get_variation_vector()
        self.variables = SystemVariables(matrices=[self.R])

    def test_variation_of_Omega_expands(self):
        """δ(Ω) → η̇ + Ω × η."""
        expr = Variation(self.Omega)
        result = apply_manifold_rules(expr, self.variables)
        assert isinstance(result, VAdd)
        assert len(result.nodes) == 2
        # First term: TimeDerivative(eta)
        assert isinstance(result.nodes[0], TimeDerivative)
        assert str(result.nodes[0].expr) == str(self.eta)
        # Second term: Cross(Omega, eta) — positive sign for SO3
        assert isinstance(result.nodes[1], Cross)
        assert str(result.nodes[1]) == str(Cross(self.Omega, self.eta))

    def test_so3_sign_differs_from_s2(self):
        """SO3 uses + Ω×η while S2 uses - ω×ξ."""
        expr = Variation(self.Omega)
        result = apply_manifold_rules(expr, self.variables)
        # Second term should be Cross directly (positive), not SVMul with -1
        assert isinstance(result.nodes[1], Cross)
        assert not isinstance(result.nodes[1], SVMul)

    def test_variation_of_eta_raises(self):
        """δ(η) is nonsensical — η is already a variation vector."""
        with pytest.raises(ValueError, match='variation vector'):
            Variation(self.eta)

    def test_nested_in_dot(self):
        """δ(Ω) inside Dot should be expanded."""
        v = Vector('v', attr=['Constant'])
        expr = Dot(Variation(self.Omega), v)
        result = apply_manifold_rules(expr, self.variables)
        assert result.has(self.eta)

    def test_omega_is_tso3(self):
        """Tangent and variation vectors should be TSO3 with parent ref."""
        assert isinstance(self.Omega, TSO3)
        assert isinstance(self.eta, TSO3)
        assert self.Omega.SO3 is self.R
        assert self.eta.SO3 is self.R

    def test_so3_delta_is_r_hat_eta(self):
        """R.delta() → R·Hat(η) (variation of rotation matrix)."""
        from geomech.core.operations.multiplication import MMMul
        result = self.R.delta()
        assert isinstance(result, MMMul)
        assert str(result.left) == str(self.R)
        assert isinstance(result.right, Hat)
        assert str(result.right.expr) == str(self.eta)

    def test_so3_t_diff_is_r_hat_omega(self):
        """R.t_diff() → R·Hat(Ω) (rotational kinematics)."""
        from geomech.core.operations.multiplication import MMMul
        result = self.R.t_diff()
        assert isinstance(result, MMMul)
        assert str(result.left) == str(self.R)
        assert isinstance(result.right, Hat)
        assert str(result.right.expr) == str(self.Omega)

    def test_nested_in_svmul(self):
        """δ(Ω) inside SVMul should be expanded."""
        s = Scalar('s', attr=['Constant'])
        expr = SVMul(Variation(self.Omega), s)
        result = apply_manifold_rules(expr, self.variables)
        assert result.has(self.eta)


# ---------------------------------------------------------------------------
# Mixed systems
# ---------------------------------------------------------------------------


class TestMixedManifoldRules:

    def test_s2_and_so3_together(self):
        """System with both S2 and SO3 manifolds."""
        q = S2('q')
        R = SO3('R')
        omega = q.get_tangent_vector()
        Omega = R.get_tangent_vector()
        xi = q.get_variation_vector()
        eta = R.get_variation_vector()

        variables = SystemVariables(vectors=[q], matrices=[R])

        # Both variations in a single expression
        expr = VAdd(Variation(omega), Variation(Omega))
        result = apply_manifold_rules(expr, variables)

        # Both should be expanded
        assert result.has(xi)
        assert result.has(eta)
        assert not any(isinstance(n, Variation) for n in result.nodes)

    def test_s2_and_plain_vector(self):
        """S2 + plain vector: only S2 tangent gets substituted."""
        q = S2('q')
        x = Vector('x')
        omega = q.get_tangent_vector()

        variables = SystemVariables(vectors=[q, x])
        expr = VAdd(Variation(omega), Variation(x))
        result = apply_manifold_rules(expr, variables)

        # omega variation expanded, x variation unchanged
        assert result.has(q.get_variation_vector())
        has_variation_x = any(
            isinstance(n, Variation) and str(n.expr) == 'x'
            for n in result.nodes)
        assert has_variation_x

    def test_two_s2_manifolds(self):
        """System with two S2 variables (e.g., double spherical pendulum)."""
        q1 = S2('q1')
        q2 = S2('q2')
        omega1 = q1.get_tangent_vector()
        omega2 = q2.get_tangent_vector()
        xi1 = q1.get_variation_vector()
        xi2 = q2.get_variation_vector()

        variables = SystemVariables(vectors=[q1, q2])
        expr = VAdd(Variation(omega1), Variation(omega2))
        result = apply_manifold_rules(expr, variables)

        # Both should be expanded independently
        assert result.has(xi1)
        assert result.has(xi2)
        assert not any(isinstance(n, Variation) for n in result.nodes)


# ---------------------------------------------------------------------------
# Composite expressions: y = x + R*q*l
# ---------------------------------------------------------------------------


class TestCompositeExpression:
    """Test delta and t_diff on y = x + R*q*l (quadrotor-payload kinematics)."""

    def setup_method(self):
        self.x = Vector('x')
        self.R = SO3('R')
        self.q = S2('q')
        self.l = Scalar('l', attr=['Constant'])
        self.Omega = self.R.get_tangent_vector()
        self.eta = self.R.get_variation_vector()
        self.omega = self.q.get_tangent_vector()
        self.xi = self.q.get_variation_vector()
        # y = x + R*q*l
        self.y = self.x + MVMul(self.R, self.q) * self.l

    def test_delta_y(self):
        """δ(y) = δx + (R·Hat(η)·q + R·(ξ×q))·l"""
        dy = self.y.delta()
        # Build expected: δx + (MMMul(R,Hat(η))*q + R*(ξ×q))*l
        expected = VAdd(
            Variation(self.x),
            SVMul(
                VAdd(
                    MVMul(MMMul(self.R, Hat(self.eta)), self.q),
                    MVMul(self.R, Cross(self.xi, self.q)),
                ),
                self.l,
            ),
        )
        assert str(dy) == str(expected)

    def test_t_diff_y(self):
        """d/dt(y) = ẋ + (R·Hat(Ω)·q + R·(ω×q))·l"""
        ydot = self.y.t_diff()
        # Build expected: ẋ + (MMMul(R,Hat(Ω))*q + R*(ω×q))*l
        expected = VAdd(
            TimeDerivative(self.x),
            SVMul(
                VAdd(
                    MVMul(MMMul(self.R, Hat(self.Omega)), self.q),
                    MVMul(self.R, Cross(self.omega, self.q)),
                ),
                self.l,
            ),
        )
        assert str(ydot) == str(expected)


# ---------------------------------------------------------------------------
# TS2 and TSO3 delta methods
# ---------------------------------------------------------------------------


class TestTangentDelta:

    def test_ts2_delta_substitute(self):
        """TS2.delta(substitute=True) → ξ̇ - ω × ξ."""
        q = S2('q')
        omega = q.get_tangent_vector()
        xi = q.get_variation_vector()
        result = omega.delta(substitute=True)
        assert isinstance(result, VAdd)
        assert result.has(xi)

    def test_ts2_delta_no_substitute(self):
        """TS2.delta() → Variation(ω)."""
        q = S2('q')
        omega = q.get_tangent_vector()
        result = omega.delta()
        assert isinstance(result, Variation)

    def test_tso3_delta_substitute(self):
        """TSO3.delta(substitute=True) → η̇ + Ω × η."""
        R = SO3('R')
        Omega = R.get_tangent_vector()
        eta = R.get_variation_vector()
        result = Omega.delta(substitute=True)
        assert isinstance(result, VAdd)
        assert result.has(eta)

    def test_tso3_delta_no_substitute(self):
        """TSO3.delta() → Variation(Ω)."""
        R = SO3('R')
        Omega = R.get_tangent_vector()
        result = Omega.delta()
        assert isinstance(result, Variation)

    def test_nested_variation_raises(self):
        """δ(δ(x)) is second-order — should raise."""
        x = Vector('x')
        with pytest.raises(ValueError, match='variation of a variation'):
            Variation(Variation(x))
