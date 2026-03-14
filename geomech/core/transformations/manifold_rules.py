"""Manifold kinematic substitutions applied during EOM derivation.

For S2 (unit sphere) with manifold point q, tangent vector ω, variation vector ξ:
  - Variation(ω) → ξ̇ - ω × ξ        (derived from δ/d/dt commutativity)
  - Variation(ξ) → left as Variation(ξ)  (handled later by IBP + extraction)

The expansion of δ(ω) follows from the commutativity of δ and d/dt on the
action integral.  Starting from q̇ = ω × q and δq = ξ × q, setting
δ(q̇) = d/dt(δq) and solving for δ(ω) gives  δ(ω) = ξ̇ - ω × ξ.
"""

from __future__ import annotations

from geomech.core.base.expressions import S2, TS2, Scalar
from geomech.core.operations.addition import VAdd
from geomech.core.operations.multiplication import SVMul
from geomech.core.operations.geometry import Cross
from geomech.core.operations.calculus import Variation, TimeDerivative

from geomech.dynamics.variables import SystemVariables


_NEG1 = Scalar('(-1)', value=-1, attr=['Constant'])


def apply_manifold_rules(expr, variables: SystemVariables):
    """Apply manifold kinematic substitutions to an expression.

    Walks the expression tree and replaces Variation(TS2) nodes with
    their kinematic expansions.
    """
    # Build lookup: tangent vector name → (S2 parent, omega, xi)
    s2_map = {}
    for vec in variables.vectors:
        if isinstance(vec, S2):
            omega = vec.get_tangent_vector()
            xi = vec.get_variation_vector()
            s2_map[str(omega)] = (vec, omega, xi)

    if not s2_map:
        return expr

    return _substitute(expr, s2_map)


def _substitute(expr, s2_map):
    """Recursively substitute manifold variations."""

    # Variation(TS2) → kinematic expansion
    if isinstance(expr, Variation):
        inner = expr.expr
        if isinstance(inner, TS2) and str(inner) in s2_map:
            q, omega, xi = s2_map[str(inner)]
            # δ(ω) = ξ̇ - ω × ξ
            return VAdd(
                TimeDerivative(xi),
                SVMul(Cross(omega, xi), _NEG1),
            )
        # Variation of non-tangent or unknown — recurse into inner
        inner_sub = _substitute(inner, s2_map)
        if inner_sub is not inner:
            return Variation(inner_sub)
        return expr

    # Recurse into children
    nodes = getattr(expr, 'nodes', None)
    if nodes is None or len(nodes) == 0:
        return expr

    new_nodes = [_substitute(n, s2_map) for n in nodes]
    if all(n is o for n, o in zip(new_nodes, nodes)):
        return expr  # nothing changed

    # Rebuild the node with new children
    return _rebuild(expr, new_nodes)


def _rebuild(expr, new_nodes):
    """Rebuild an expression node with new children."""
    from geomech.core.operations.addition import Add, VAdd, MAdd
    from geomech.core.operations.multiplication import (
        Mul, SVMul, SMMul, MVMul, MMMul, VVMul,
    )
    from geomech.core.operations.geometry import Dot, Cross, Hat, Vee, Transpose
    from geomech.core.operations.calculus import Variation, TimeDerivative, TimeIntegral

    cls = type(expr)

    # N-ary (Add, VAdd, MAdd)
    if isinstance(expr, (Add, VAdd, MAdd)):
        return cls(*new_nodes)

    # Binary ops
    if isinstance(expr, (Mul, SVMul, SMMul, MVMul, MMMul, VVMul, Dot, Cross)):
        return cls(new_nodes[0], new_nodes[1])

    # Unary ops
    if isinstance(expr, (Hat, Vee, Transpose, Variation, TimeDerivative, TimeIntegral)):
        return cls(new_nodes[0])

    return expr
