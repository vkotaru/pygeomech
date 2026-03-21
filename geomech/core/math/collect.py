"""Collect: rearrange a scalar expression so that a target vector
appears on the left side of Dot.

This is a preprocessing step for integration by parts (ibp), which
only matches ``Dot(target, ...)`` on the left.

Uses:
  - Dot commutativity: Dot(a, b) = Dot(b, a)
  - Scalar triple product: Dot(a, Cross(b, c)) = Dot(b, Cross(c, a))
  - Orthogonal invariance: Dot(R*a, R*b) = Dot(a, b) for R in SO3
"""

from __future__ import annotations

from geomech.core.base.expressions import SO3
from geomech.core.operations.addition import Add
from geomech.core.operations.multiplication import Mul, MVMul, MMMul
from geomech.core.operations.geometry import Dot, Cross, Transpose


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

        # Both sides are MVMul with the same rotation matrix prefix
        # Dot(R*a, R*b) = Dot(a, b) for orthogonal R (SO3)
        case (MVMul(), MVMul()):
            if (isinstance(l.left, SO3) and isinstance(r.left, SO3)
                    and l.left == r.left):
                # Strip the common rotation: Dot(R*a, R*b) → Dot(a, b)
                return collect(Dot(l.right, r.right), vec)
            # General MVMul case: try to expose vec
            if l.right == vec:
                # Dot(M*vec, w) → Dot(vec, M^T*w)
                return Dot(l.right, MVMul(Transpose(l.left), r))
            if r.right == vec:
                return Dot(r.right, MVMul(Transpose(r.left), l))
            # Recurse: if vec is deeper, try flipping to put the MVMul
            # containing vec on the left
            if l.has(vec) and not r.has(vec):
                pass  # l has vec, fall through to MVMul patterns below
            elif r.has(vec) and not l.has(vec):
                return collect(Dot(r, l), vec)
            return expr

        # MVMul on one side only
        case (MVMul(), _):
            if l.right == vec:
                # Dot(M*vec, w) → Dot(vec, M^T*w)
                return Dot(l.right, MVMul(Transpose(l.left), r))
            # vec is deeper in the MVMul — try to pull it out
            if l.has(vec):
                inner = l.right
                if isinstance(inner, Cross):
                    # Dot(M*Cross(vec, b), w) — use triple product
                    # Dot(M*(vec×b), w) = (M*(vec×b))^T * w = (vec×b)^T * M^T * w
                    # = Dot(vec×b, M^T*w) = Dot(vec, Cross(b, M^T*w))
                    if inner.left == vec:
                        return Dot(vec, Cross(inner.right, MVMul(Transpose(l.left), r)))
                    if inner.right == vec:
                        return Dot(vec, Cross(MVMul(Transpose(l.left), r), inner.left))
            # vec not in MVMul — flip and let other patterns try
            return collect(Dot(r, l), vec)

        case (_, MVMul()):
            # Flip so MVMul is on the left, recurse
            return collect(Dot(r, l), vec)

        # Cross on one or both sides — scalar triple product
        case (Cross(), _) if l.has(vec):
            # Flip so vec's Cross is on the right, then use triple product
            return collect(Dot(r, l), vec)

        case (_, Cross()):
            if r.left == vec:
                # Dot(a, Cross(vec, c)) = Dot(vec, Cross(c, a))
                return Dot(vec, Cross(r.right, l))
            if r.right == vec:
                # Dot(a, Cross(b, vec)) = Dot(vec, Cross(a, b))
                return Dot(vec, Cross(l, r.left))
            # vec is nested deeper in the Cross (e.g. Cross(f(vec), b))
            if r.left.has(vec):
                # Dot(a, Cross(f(vec), c)) — use triple product
                # = Dot(f(vec), Cross(c, a))
                return collect(Dot(r.left, Cross(r.right, l)), vec)
            if r.right.has(vec):
                # Dot(a, Cross(b, f(vec))) = Dot(f(vec), Cross(a, b))
                return collect(Dot(r.right, Cross(l, r.left)), vec)
            return expr

        case _:
            return expr
