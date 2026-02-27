"""
simplify(expr)      — pull scalars, apply vector identities, eliminate
                       zeros, combine numeric constants.
full_simplify(expr) — expand once, then loop simplify until convergence.
"""

from __future__ import annotations

from geomech.core.base.expressions import (
    Scalar, Zero, ZeroVector, ZeroMatrix,
)
from geomech.core.base.types import ExprType
from geomech.core.operations.addition import Add, VAdd, MAdd
from geomech.core.operations.multiplication import (
    Mul, SVMul, SMMul, MVMul, MMMul, VVMul,
)
from geomech.core.operations.geometry import Dot, Cross, Hat, Vee, Transpose
from geomech.core.operations.calculus import Variation, TimeDerivative, TimeIntegral


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def simplify(expr):
    """Pull scalars, apply vector identities, then eliminate zeros and
    combine numeric constants — single call does everything."""
    from geomech.core.transformations.pull import pull
    from geomech.core.transformations.vector_rules import vector_rules

    expr = pull(expr)
    expr = vector_rules(expr)
    return _eliminate(expr)


def full_simplify(expr, max_iter=10):
    """Expand once, then loop simplify until convergence.

    Parameters
    ----------
    max_iter : int
        Safety cap on the number of passes (default 10).
    """
    from geomech.core.transformations.expand import expand

    expr = expand(expr)
    for _ in range(max_iter):
        prev = expr
        expr = simplify(expr)
        if str(expr) == str(prev):
            break
    return expr


# ---------------------------------------------------------------------------
# Bottom-up zero / identity / constant elimination
# ---------------------------------------------------------------------------

def _eliminate(expr):
    """Bottom-up pass: zeros, identities, numeric constant combining."""
    match expr:
        # ---- n-ary addition ----
        case Add(nodes=nodes):
            return _simplify_add([_eliminate(n) for n in nodes])
        case VAdd(nodes=nodes):
            return _simplify_vadd([_eliminate(n) for n in nodes])
        case MAdd(nodes=nodes):
            return _simplify_madd([_eliminate(n) for n in nodes])

        # ---- scalar * scalar ----
        case Mul():
            return _simplify_mul(_eliminate(expr.left), _eliminate(expr.right))

        # ---- scalar * vector ----
        case SVMul():
            vec, scl = _eliminate(expr.left), _eliminate(expr.right)
            if _is_zero(scl) or _is_zero(vec):
                return ZeroVector
            if _is_one(scl):
                return vec
            return SVMul(vec, scl)

        # ---- scalar * matrix ----
        case SMMul():
            mat, scl = _eliminate(expr.left), _eliminate(expr.right)
            if _is_zero(scl) or _is_zero(mat):
                return ZeroMatrix
            if _is_one(scl):
                return mat
            return SMMul(mat, scl)

        # ---- matrix * vector ----
        case MVMul():
            mat, vec = _eliminate(expr.left), _eliminate(expr.right)
            if _is_zero(mat) or _is_zero(vec):
                return ZeroVector
            if _is_identity(mat):
                return vec
            return MVMul(mat, vec)

        # ---- matrix * matrix ----
        case MMMul():
            l, r = _eliminate(expr.left), _eliminate(expr.right)
            if _is_zero(l) or _is_zero(r):
                return ZeroMatrix
            if _is_identity(l):
                return r
            if _is_identity(r):
                return l
            return MMMul(l, r)

        # ---- vector * vector ----
        case VVMul():
            l, r = _eliminate(expr.left), _eliminate(expr.right)
            if _is_zero(l) or _is_zero(r):
                return Zero
            return VVMul(l, r)

        # ---- binary geometry ops ----
        case Dot():
            l, r = _eliminate(expr.left), _eliminate(expr.right)
            if _is_zero(l) or _is_zero(r):
                return Zero
            return Dot(l, r)

        case Cross():
            l, r = _eliminate(expr.left), _eliminate(expr.right)
            if _is_zero(l) or _is_zero(r):
                return ZeroVector
            return Cross(l, r)

        # ---- unary geometry ops ----
        case Hat():
            inner = _eliminate(expr.expr)
            if _is_zero(inner):
                return ZeroMatrix
            if isinstance(inner, Vee):
                return inner.expr
            return Hat(inner)

        case Vee():
            inner = _eliminate(expr.expr)
            if _is_zero(inner):
                return ZeroVector
            if isinstance(inner, Hat):
                return inner.expr
            return Vee(inner)

        case Transpose():
            inner = _eliminate(expr.expr)
            if isinstance(inner, Transpose):
                return inner.expr
            return Transpose(inner)

        # ---- calculus ops ----
        case Variation():
            inner = _eliminate(expr.expr)
            if _is_zero(inner):
                return _zero_for(inner)
            return Variation(inner)

        case TimeDerivative():
            inner = _eliminate(expr.expr)
            if _is_zero(inner):
                return _zero_for(inner)
            return TimeDerivative(inner)

        case TimeIntegral():
            inner = _eliminate(expr.expr)
            if _is_zero(inner):
                return _zero_for(inner)
            return TimeIntegral(inner)

        # ---- leaves ----
        case _:
            return expr


# ---------------------------------------------------------------------------
# Predicates
# ---------------------------------------------------------------------------

def _is_zero(expr) -> bool:
    return getattr(expr, 'isZero', False)


def _is_one(expr) -> bool:
    return getattr(expr, 'value', None) == 1


def _is_identity(expr) -> bool:
    flags = getattr(expr, '_flags', None)
    return flags is not None and flags.is_identity


def _is_numeric_leaf(expr) -> bool:
    """True for leaf Scalars with a numeric value (not operation nodes)."""
    return (isinstance(expr, Scalar)
            and not hasattr(expr, 'nodes')
            and expr.value is not None)


def _zero_for(expr):
    """Return the appropriate typed zero constant."""
    match getattr(expr, 'type', None):
        case ExprType.VECTOR:
            return ZeroVector
        case ExprType.MATRIX:
            return ZeroMatrix
        case _:
            return Zero


# ---------------------------------------------------------------------------
# Per-node helpers
# ---------------------------------------------------------------------------

def _simplify_add(nodes):
    """Scalar addition: filter zeros, combine numerics, unwrap single."""
    numeric_total = 0
    kept = []
    for n in nodes:
        if _is_zero(n):
            continue
        if _is_numeric_leaf(n):
            numeric_total += n.value
        else:
            kept.append(n)
    if numeric_total != 0:
        kept.append(Scalar('(' + str(numeric_total) + ')',
                           value=numeric_total, attr=['Constant']))
    if len(kept) == 0:
        return Zero
    if len(kept) == 1:
        return kept[0]
    return Add(*kept)


def _simplify_vadd(nodes):
    """Vector addition: filter zeros, unwrap single."""
    kept = [n for n in nodes if not _is_zero(n)]
    if len(kept) == 0:
        return ZeroVector
    if len(kept) == 1:
        return kept[0]
    return VAdd(*kept)


def _simplify_madd(nodes):
    """Matrix addition: filter zeros, unwrap single."""
    kept = [n for n in nodes if not _is_zero(n)]
    if len(kept) == 0:
        return ZeroMatrix
    if len(kept) == 1:
        return kept[0]
    return MAdd(*kept)


def _simplify_mul(l, r):
    """Scalar multiplication: zeros, identity, numeric combining."""
    if _is_zero(l) or _is_zero(r):
        return Zero
    if _is_one(l):
        return r
    if _is_one(r):
        return l
    # both numeric leaves → combine
    if _is_numeric_leaf(l) and _is_numeric_leaf(r):
        combined = l.value * r.value
        return Scalar('(' + str(combined) + ')',
                       value=combined, attr=['Constant'])
    return Mul(l, r)
