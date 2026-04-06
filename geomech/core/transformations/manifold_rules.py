"""Manifold kinematic substitutions applied during EOM derivation.

For S2 (unit sphere) with manifold point q, tangent vector ω, variation vector ξ:
  - Variation(ω) → ξ̇ - ω × ξ        (derived from δ/d/dt commutativity)

For SO3 (rotation group) with tangent vector Ω, variation vector η:
  - Variation(Ω) → η̇ + Ω × η        (body-frame angular velocity convention)

The expansion of δ(ω) follows from the commutativity of δ and d/dt on the
action integral.  Starting from q̇ = ω × q and δq = ξ × q, setting
δ(q̇) = d/dt(δq) and solving for δ(ω) gives  δ(ω) = ξ̇ - ω × ξ.

Similarly, for SO3 with Ṙ = R·Hat(Ω) and δR = R·Hat(η), commutativity
of δ and d/dt gives  δ(Ω) = η̇ + Ω × η.
"""

from __future__ import annotations

from geomech.core.base.expressions import S2, SO3, TS2, TSO3, Scalar
from geomech.core.operations.addition import VAdd
from geomech.core.operations.calculus import TimeDerivative, Variation
from geomech.core.operations.geometry import Cross
from geomech.core.operations.multiplication import SVMul
from geomech.dynamics.variables import SystemVariables

_NEG1 = Scalar("(-1)", value=-1, attr=["Constant"])


def apply_manifold_rules(expr, variables: SystemVariables):
    """Apply manifold kinematic substitutions to an expression.

    Walks the expression tree and replaces Variation(TS2) and
    Variation(TSO3) nodes with their kinematic expansions.
    """
    tangent_map = {}

    # S2 manifolds (vectors)
    for vec in variables.vectors:
        if isinstance(vec, S2):
            omega = vec.get_tangent_vector()
            xi = vec.get_variation_vector()
            tangent_map[str(omega)] = ("S2", vec, omega, xi)

    # SO3 manifolds (matrices)
    for mat in variables.matrices:
        if isinstance(mat, SO3):
            Omega = mat.get_tangent_vector()
            eta = mat.get_variation_vector()
            tangent_map[str(Omega)] = ("SO3", mat, Omega, eta)

    if not tangent_map:
        return expr

    return _substitute(expr, tangent_map)


def _substitute(expr, tangent_map):
    """Recursively substitute manifold variations."""

    # Variation(TS2) or Variation(TSO3) → kinematic expansion
    if isinstance(expr, Variation):
        inner = expr.expr
        if isinstance(inner, (TS2, TSO3)) and str(inner) in tangent_map:
            kind, parent, omega, var_vec = tangent_map[str(inner)]
            if kind == "S2":
                # δ(ω) = ξ̇ - ω × ξ
                return VAdd(
                    TimeDerivative(var_vec),
                    SVMul(Cross(omega, var_vec), _NEG1),
                )
            elif kind == "SO3":
                # δ(Ω) = η̇ + Ω × η
                return VAdd(
                    TimeDerivative(var_vec),
                    Cross(omega, var_vec),
                )
            else:
                raise ValueError(f"Unknown manifold kind: {kind}")
        # Variation of non-tangent or unknown — recurse into inner
        inner_sub = _substitute(inner, tangent_map)
        if inner_sub is not inner:
            return Variation(inner_sub)
        return expr

    # Recurse into children
    nodes = getattr(expr, "nodes", None)
    if nodes is None or len(nodes) == 0:
        return expr

    new_nodes = [_substitute(n, tangent_map) for n in nodes]
    if all(n is o for n, o in zip(new_nodes, nodes)):
        return expr  # nothing changed

    # Rebuild the node with new children
    return _rebuild(expr, new_nodes)


def _rebuild(expr, new_nodes):
    """Rebuild an expression node with new children."""
    from geomech.core.operations.addition import Add, MAdd, VAdd
    from geomech.core.operations.calculus import TimeDerivative, TimeIntegral, Variation
    from geomech.core.operations.geometry import Cross, Dot, Hat, Transpose, Vee
    from geomech.core.operations.multiplication import (
        MMMul,
        Mul,
        MVMul,
        SMMul,
        SVMul,
        VVMul,
    )

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
