"""Collect: rearrange a scalar expression so that a target vector
appears on the left side of Dot.

This is a preprocessing step for integration by parts (ibp), which
only matches ``Dot(target, ...)`` on the left.

Uses:
  - Dot commutativity: Dot(a, b) = Dot(b, a)
  - Scalar triple product: Dot(a, Cross(b, c)) = Dot(b, Cross(c, a))
"""

from __future__ import annotations

from geomech.core.operations.addition import Add
from geomech.core.operations.multiplication import Mul, MVMul
from geomech.core.operations.geometry import Dot, Cross


def collect(expr, vec):
    """Collect *expr* with respect to *vec*.

    Rearranges so *vec* appears on the left of every Dot that contains it.
    """
    match expr:
        case Add(nodes=nodes):
            return Add(*[collect(n, vec) for n in nodes])

        case Mul():
            return Mul(collect(expr.left, vec), collect(expr.right, vec))

        case Dot():
            return _collect_dot(expr, vec)

        case _:
            return expr


def _collect_dot(expr, vec):
    """Rearrange a Dot so *vec* is on the left."""
    l, r = expr.left, expr.right

    # Already canonical
    if l == vec:
        return expr

    # Simple flip
    if r == vec:
        return Dot(r, l)

    # Neither side contains vec — nothing to do
    if not l.has(vec) and not r.has(vec):
        return expr

    # --- structural patterns (at least one side has vec) ---

    match (l, r):
        # MVMul + Cross: kinetic energy patterns
        case (Cross(), MVMul()):
            # Flip so MVMul is on the left, then recurse
            return collect(Dot(r, l), vec)

        case (MVMul(), Cross()):
            # Dot(M*a, Cross(b, c)) — use triple product to put vec on left
            if r.left == vec:
                # Dot(M*a, Cross(vec, c)) = Dot(vec, Cross(c, M*a))
                return Dot(vec, Cross(r.right, l))
            if r.right == vec:
                # Dot(M*a, Cross(b, vec)) = Dot(vec, Cross(M*a, b))
                return Dot(vec, Cross(l, r.left))
            if l.right == vec:
                # Dot(M*vec, Cross(b, c)) = Dot(vec, Cross(M*b, c))
                return Dot(vec, Cross(MVMul(l.left, r.left), r.right))
            return expr

        # MVMul on one side only
        case (MVMul(), _):
            if l.right == vec:
                # Dot(M*vec, w) → Dot(vec, M*w)
                return Dot(l.right, MVMul(l.left, r))
            # vec not in MVMul.right — flip and let other patterns try
            return Dot(r, l)

        case (_, MVMul()):
            # Flip so MVMul is on the left, recurse
            return collect(Dot(r, l), vec)

        # Cross on the right — scalar triple product
        case (_, Cross()):
            if r.left == vec:
                # Dot(a, Cross(vec, c)) = Dot(vec, Cross(c, a))
                return Dot(vec, Cross(r.right, l))
            if r.right == vec:
                # Dot(a, Cross(b, vec)) = Dot(vec, Cross(a, b))
                return Dot(vec, Cross(l, r.left))
            return expr

        case _:
            return expr
