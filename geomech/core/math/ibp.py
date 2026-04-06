"""Integration by parts for scalar expressions.

    integrate_by_parts(expr, targets) — full pipeline: expand → simplify → collect → IBP per target

IBP identity (boundary terms dropped):
    ∫ Dot(ẋ, f) dt  =  -∫ Dot(x, ḟ) dt

So:  Dot(target, rhs)  →  Dot(-target.t_integrate(), rhs.t_diff())
"""

from __future__ import annotations

from geomech.core.math.collect import collect
from geomech.core.operations.addition import Add
from geomech.core.operations.geometry import Dot
from geomech.core.operations.multiplication import Mul
from geomech.core.transformations.expand import expand
from geomech.core.transformations.simplify import full_simplify

# ---------------------------------------------------------------------------
# Single IBP pass
# ---------------------------------------------------------------------------


def _apply_ibp(expr, target):
    """Apply integration by parts to *expr* with respect to *target*.

    Recurses through Add and Mul.  At each Dot where the left side
    equals *target*, applies the IBP identity.
    """
    match expr:
        case Add(nodes=nodes):
            return Add(*[_apply_ibp(n, target) for n in nodes])

        case Mul():
            return Mul(_apply_ibp(expr.left, target), _apply_ibp(expr.right, target))

        case Dot():
            if expr.left == target:
                return Dot(expr.left.t_integrate() * (-1), expr.right.t_diff())
            return expr

        case _:
            return expr


# ---------------------------------------------------------------------------
# Full IBP pipeline
# ---------------------------------------------------------------------------


def integrate_by_parts(expr, targets):
    """Expand, simplify, then apply collect + IBP for each target.

    *targets* is a list of variation vector time-derivatives (e.g. dot_eta)
    that should be integrated by parts to remove the time derivative.
    """
    expr = expand(expr)
    expr = full_simplify(expr)
    for target in targets:
        expr = collect(expr, target)
        expr = _apply_ibp(expr, target)
        expr = full_simplify(expr)
    return expr
