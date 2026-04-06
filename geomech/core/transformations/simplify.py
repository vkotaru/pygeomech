"""
simplify(expr)      — pull scalars, apply vector identities, eliminate
                       zeros, combine numeric constants.
full_simplify(expr) — expand once, then loop simplify until convergence.
"""

from __future__ import annotations

from geomech.core.base.expressions import (
    IdentityMatrix,
    Scalar,
    Zero,
    ZeroMatrix,
    ZeroVector,
)
from geomech.core.base.types import ExprType
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
            # Hat(v) * w = Cross(v, w)
            if isinstance(mat, Hat):
                return Cross(mat.expr, vec)
            # (A * B) * v → A * (B * v)  (matrix associativity)
            if isinstance(mat, MMMul):
                return _eliminate(MVMul(mat.left, MVMul(mat.right, vec)))
            # A * (B * v) → (A*B) * v  when A*B simplifies (e.g. R'*R = I)
            if isinstance(vec, MVMul):
                combined = _eliminate(MMMul(mat, vec.left))
                if not isinstance(combined, MMMul):
                    # A*B simplified (e.g. to I), so use the result
                    return _eliminate(MVMul(combined, vec.right))
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
            # R^T * R = I  when R is orthogonal (SO3)
            if isinstance(l, Transpose) and l.expr.is_orthogonal and l.expr == r:
                return IdentityMatrix
            # R * R^T = I  when R is orthogonal (SO3)
            if isinstance(r, Transpose) and r.expr.is_orthogonal and r.expr == l:
                return IdentityMatrix
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
            # Symmetric matrix: J^T = J
            if inner.is_symmetric:
                return inner
            # Hat is skew-symmetric: Hat(v)^T = -Hat(v)
            if isinstance(inner, Hat):
                return SMMul(inner, Scalar("(-1)", value=-1, attr=["Constant"]))
            # (s*M)^T = s*M^T
            if isinstance(inner, SMMul):
                return SMMul(_eliminate(Transpose(inner.left)), inner.right)
            # (A*B)^T = B^T * A^T
            if isinstance(inner, MMMul):
                return _eliminate(MMMul(Transpose(inner.right), Transpose(inner.left)))
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
            # d/dt(∫x dt) = x
            if isinstance(inner, TimeIntegral):
                return inner.expr
            return TimeDerivative(inner)

        case TimeIntegral():
            inner = _eliminate(expr.expr)
            if _is_zero(inner):
                return _zero_for(inner)
            # ∫(d/dt(x)) dt = x
            if isinstance(inner, TimeDerivative):
                return inner.expr
            return TimeIntegral(inner)

        # ---- leaves ----
        case _:
            return expr


# ---------------------------------------------------------------------------
# Predicates
# ---------------------------------------------------------------------------


def _is_zero(expr) -> bool:
    return getattr(expr, "is_zero", False)


def _is_one(expr) -> bool:
    return getattr(expr, "value", None) == 1


def _is_identity(expr) -> bool:
    flags = getattr(expr, "flags", None)
    return flags is not None and flags.is_identity


def _is_numeric_leaf(expr) -> bool:
    """True for leaf Scalars with a numeric value (not operation nodes)."""
    return isinstance(expr, Scalar) and not hasattr(expr, "nodes") and expr.value is not None


def _zero_for(expr):
    """Return the appropriate typed zero constant."""
    match getattr(expr, "type", None):
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
    """Scalar addition: filter zeros, combine numerics, collect like terms."""
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
        kept.append(Scalar("(" + str(numeric_total) + ")", value=numeric_total, attr=["Constant"]))
    kept = _collect_like_terms(kept, ExprType.SCALAR)
    if len(kept) == 0:
        return Zero
    if len(kept) == 1:
        return kept[0]
    return Add(*kept)


def _simplify_vadd(nodes):
    """Vector addition: filter zeros, collect like terms."""
    kept = [n for n in nodes if not _is_zero(n)]
    kept = _collect_like_terms(kept, ExprType.VECTOR)
    if len(kept) == 0:
        return ZeroVector
    if len(kept) == 1:
        return kept[0]
    return VAdd(*kept)


def _simplify_madd(nodes):
    """Matrix addition: filter zeros, collect like terms."""
    kept = [n for n in nodes if not _is_zero(n)]
    kept = _collect_like_terms(kept, ExprType.MATRIX)
    if len(kept) == 0:
        return ZeroMatrix
    if len(kept) == 1:
        return kept[0]
    return MAdd(*kept)


# ---------------------------------------------------------------------------
# Like-term collection
# ---------------------------------------------------------------------------


def _split_coeff(expr):
    """Split expr into (numeric_coefficient, term).

    For Mul(term, Number) or Mul(Number, term), returns (number, term).
    For SVMul(vec, Number) or SMMul(mat, Number), returns (number, vec/mat).
    Otherwise returns (1, expr).
    """
    if isinstance(expr, Mul):
        if _is_numeric_leaf(expr.left):
            return expr.left.value, expr.right
        if _is_numeric_leaf(expr.right):
            return expr.right.value, expr.left
    if isinstance(expr, SVMul):
        if _is_numeric_leaf(expr.right):
            return expr.right.value, expr.left
    if isinstance(expr, SMMul):
        if _is_numeric_leaf(expr.right):
            return expr.right.value, expr.left
    if _is_numeric_leaf(expr):
        return expr.value, None
    return 1, expr


def _commutative_equal(a, b):
    """Structural equality that respects commutativity of Dot and Mul."""
    if a is b:
        return True
    if type(a) is not type(b):
        return False

    # Commutative binary ops: check both orderings
    if isinstance(a, Dot):
        return (_commutative_equal(a.left, b.left) and _commutative_equal(a.right, b.right)) or (
            _commutative_equal(a.left, b.right) and _commutative_equal(a.right, b.left)
        )
    if isinstance(a, Mul):
        return (_commutative_equal(a.left, b.left) and _commutative_equal(a.right, b.right)) or (
            _commutative_equal(a.left, b.right) and _commutative_equal(a.right, b.left)
        )

    # Non-commutative binary ops: order matters
    if isinstance(a, (SVMul, SMMul, MVMul, MMMul, VVMul, Cross)):
        return _commutative_equal(a.left, b.left) and _commutative_equal(a.right, b.right)

    # N-ary: same length and all children match in order
    a_nodes = getattr(a, "nodes", None)
    b_nodes = getattr(b, "nodes", None)
    if a_nodes is not None and b_nodes is not None:
        if len(a_nodes) != len(b_nodes):
            return False
        return all(_commutative_equal(an, bn) for an, bn in zip(a_nodes, b_nodes))

    # Unary ops
    a_inner = getattr(a, "expr", None)
    b_inner = getattr(b, "expr", None)
    if a_inner is not None and b_inner is not None:
        return _commutative_equal(a_inner, b_inner)

    # Leaves: fall back to string comparison
    return str(a) == str(b)


def _make_term(coeff, term, expr_type):
    """Reconstruct coeff * term for the given type."""
    if term is None:
        # Pure numeric
        return Scalar("(" + str(coeff) + ")", value=coeff, attr=["Constant"])
    if coeff == 0:
        return None
    if coeff == 1:
        return term
    scalar = Scalar("(" + str(coeff) + ")", value=coeff, attr=["Constant"])
    if expr_type == ExprType.VECTOR:
        return SVMul(term, scalar)
    if expr_type == ExprType.MATRIX:
        return SMMul(term, scalar)
    return Mul(term, scalar)


def _collect_like_terms(nodes, expr_type):
    """Group terms by commutative equality and sum their coefficients."""
    if len(nodes) <= 1:
        return nodes

    # Split each node into (coeff, term)
    pairs = [_split_coeff(n) for n in nodes]

    # Group by commutative equality
    merged_coeffs = list(range(len(pairs)))  # index of canonical representative
    coeffs = [c for c, _ in pairs]
    terms = [t for _, t in pairs]

    for i in range(len(pairs)):
        if merged_coeffs[i] != i:
            continue  # already merged into an earlier group
        for j in range(i + 1, len(pairs)):
            if merged_coeffs[j] != j:
                continue  # already merged
            ti = terms[i]
            tj = terms[j]
            # Both pure numeric (term is None) — already combined in _simplify_add
            if ti is None and tj is None:
                continue
            if ti is not None and tj is not None and _commutative_equal(ti, tj):
                coeffs[i] += coeffs[j]
                coeffs[j] = 0
                merged_coeffs[j] = i

    # Reconstruct
    result = []
    for i in range(len(pairs)):
        if merged_coeffs[i] != i:
            continue
        made = _make_term(coeffs[i], terms[i], expr_type)
        if made is not None:
            result.append(made)
    return result


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
        return Scalar("(" + str(combined) + ")", value=combined, attr=["Constant"])
    # Fold numeric constants through nested Muls:
    # e.g. Mul(-0.5, Mul(m, 2)) → Mul(m, -1.0)
    result = Mul(l, r)
    factors = []
    _flatten_mul(result, factors)
    numerics = [f for f in factors if _is_numeric_leaf(f)]
    if len(numerics) >= 2:
        others = [f for f in factors if not _is_numeric_leaf(f)]
        combined = 1
        for n in numerics:
            combined *= n.value
        if combined == 0:
            return Zero
        if combined != 1:
            others.append(Scalar("(" + str(combined) + ")", value=combined, attr=["Constant"]))
        if len(others) == 0:
            return Scalar("(" + str(combined) + ")", value=combined, attr=["Constant"])
        # Rebuild left-associative chain
        out = others[0]
        for o in others[1:]:
            out = Mul(out, o)
        return out
    return result


def _flatten_mul(expr, factors):
    """Flatten nested Mul into a list of factors."""
    if isinstance(expr, Mul):
        _flatten_mul(expr.left, factors)
        _flatten_mul(expr.right, factors)
    else:
        factors.append(expr)
