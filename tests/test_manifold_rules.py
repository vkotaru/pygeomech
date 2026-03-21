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
    Scalar, Vector, Matrix, S2, SO3, TS2, TSO3, Zero, ZeroVector,
)
from geomech.core.operations.addition import VAdd
from geomech.core.operations.multiplication import SVMul, MVMul
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

    def test_variation_of_xi_unchanged(self):
        """δ(ξ) should not be substituted (ξ is variation, not tangent)."""
        expr = Variation(self.xi)
        result = apply_manifold_rules(expr, self.variables)
        # xi is a TS2 but not in the tangent_map (only omega is)
        assert isinstance(result, Variation)

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

    def test_so3_sign_differs_from_s2(self):
        """SO3 uses + Ω×η while S2 uses - ω×ξ."""
        expr = Variation(self.Omega)
        result = apply_manifold_rules(expr, self.variables)
        # Second term should be Cross directly (positive), not SVMul with -1
        assert isinstance(result.nodes[1], Cross)
        assert not isinstance(result.nodes[1], SVMul)

    def test_variation_of_eta_unchanged(self):
        """δ(η) should not be substituted."""
        expr = Variation(self.eta)
        result = apply_manifold_rules(expr, self.variables)
        assert isinstance(result, Variation)

    def test_nested_in_dot(self):
        """δ(Ω) inside Dot should be expanded."""
        v = Vector('v', attr=['Constant'])
        expr = Dot(Variation(self.Omega), v)
        result = apply_manifold_rules(expr, self.variables)
        assert result.has(self.eta)

    def test_omega_is_tso3(self):
        """Tangent and variation vectors should be TSO3."""
        assert isinstance(self.Omega, TSO3)
        assert isinstance(self.eta, TSO3)
        assert self.Omega.SO3 is self.R
        assert self.eta.SO3 is self.R


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
            for n in result.nodes
        )
        assert has_variation_x


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
