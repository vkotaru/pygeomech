"""Algebraic identities for vector/matrix geometry operations.

Rules applied:
  - cross(x, x) = 0              self-cross is zero vector
  - cross(x, 0) = cross(0, x) = 0  zero absorption
  - dot(x, x) = 1                when x has unit norm
  - dot(x, 0) = dot(0, x) = 0    zero absorption
  - dot(x, cross(x, y)) = 0      orthogonality
  - dot(x, cross(y, x)) = 0      orthogonality
  - dot(x, cross(y, y)) = 0      self-cross is zero
  - Same rules when cross is on the left side of dot (commutativity)
  - dot(ω, q) = 0                tangent vector orthogonal to S2 manifold point
  - cross(q, cross(ω, q)) = ω    S2 tangent recovery (unit norm + orthogonality)
  - Transpose(Transpose(x)) = x  double transpose cancellation
  - Hat(0) = ZeroMatrix           hat of zero vector
  - Linearity through Add and Mul
"""

from geomech.core.base.expressions import S2, TS2, Scalar, ZeroMatrix, ZeroVector
from geomech.core.operations.addition import Add, VAdd
from geomech.core.operations.geometry import Cross, Dot, Hat, Transpose
from geomech.core.operations.multiplication import Mul, SVMul


def _is_orthogonal_dot_cross(vec, cross_expr):
    """Check if dot(vec, cross(a, b)) is zero by orthogonality.

    Returns True when:
      - vec == a or vec == b  (x perpendicular to x x y)
      - a == b                (x x x = 0)
    """
    return vec == cross_expr.left or vec == cross_expr.right or cross_expr.left == cross_expr.right


def _is_tangent_orthogonal(a, b):
    """Check if dot(a, b) = 0 due to manifold tangent-point orthogonality.

    On S2: tangent vectors (ω, ξ) are orthogonal to the manifold point q.
    Returns True when one is a TS2 and the other is its parent S2.
    """
    if isinstance(a, TS2) and isinstance(b, S2) and a.S2 is b:
        return True
    if isinstance(b, TS2) and isinstance(a, S2) and b.S2 is a:
        return True
    return False


def _try_s2_cross_reduction(expr):
    """Simplify Cross expressions using S2 unit-norm + tangent orthogonality.

    BAC-CAB: a × (b × c) = b(a·c) - c(a·b)

    When q is S2 (‖q‖=1) and a is known tangent (a ⊥ q):
      q × (a × q) = a(q·q) - q(q·a) = a

    Only fires when tangency is structurally provable (direct TS2 or
    Cross(*, q)). For other cases (e.g. d/dt(ω)), the constraint
    system (see GitHub issue #3) is needed.
    """
    l, r = expr.left, expr.right
    _neg1 = Scalar("(-1)", value=-1, attr=["Constant"])

    # Pattern: Cross(q, Cross(a, q)) where q is S2, a is tangent to q
    if isinstance(l, S2) and isinstance(r, Cross) and r.right == l:
        a = r.left
        if _is_tangent_to_s2(a, l):
            return a

    # Pattern: Cross(q, Cross(q, a)) = -a (when a is tangent)
    if isinstance(l, S2) and isinstance(r, Cross) and r.left == l:
        a = r.right
        if _is_tangent_to_s2(a, l):
            return SVMul(a, _neg1)

    # Pattern: Cross(Cross(q, a), q) = a (when a is tangent)
    if isinstance(l, Cross) and isinstance(r, S2) and l.left == r:
        a = l.right
        if _is_tangent_to_s2(a, r):
            return a

    # Pattern: Cross(Cross(a, q), q) = -a (when a is tangent)
    if isinstance(l, Cross) and isinstance(r, S2) and l.right == r:
        a = l.left
        if _is_tangent_to_s2(a, r):
            return SVMul(a, _neg1)

    # Pattern: Cross(a, Cross(a, q)) = -q*(a·a) when a is tangent to q
    if isinstance(r, Cross) and r.right is not None and isinstance(r.right, S2) and l == r.left:
        q = r.right
        a = l
        if _is_tangent_to_s2(a, q):
            return SVMul(q, Mul(Dot(a, a), _neg1))

    return None


def _is_tangent_to_s2(expr, s2):
    """Check if expr is a tangent vector to the given S2 manifold.

    Returns True for TS2 vectors whose parent is the given S2,
    or for Cross(anything, q) which is perpendicular to q by definition.

    Does NOT recurse into TimeDerivative or Variation — while ω̇ ⊥ q
    is provable for S2, this requires a constraint system to verify
    properly (see GitHub issue #3). Without constraints, we only
    apply BAC-CAB to vectors we can structurally confirm as tangent.
    """
    # Direct TS2
    if isinstance(expr, TS2) and expr.S2 is s2:
        return True
    # Cross(a, q) is tangent to q (perpendicular to q by definition)
    if isinstance(expr, Cross):
        if expr.right == s2 or expr.left == s2:
            return True
    return False


_ZERO = lambda: Scalar("0", value=0, attr=["Constant", "Zero"])
_ONE = lambda: Scalar("1", value=1, attr=["Constant", "Ones"])


def vector_rules(expr):
    match expr:
        # --- Linearity through scalar and vector ops ---
        case Add(nodes=nodes):
            return Add(*[vector_rules(n) for n in nodes])

        case Mul():
            return Mul(vector_rules(expr.left), vector_rules(expr.right))

        case VAdd(nodes=nodes):
            return VAdd(*[vector_rules(n) for n in nodes])

        case SVMul():
            return SVMul(vector_rules(expr.left), vector_rules(expr.right))

        # --- Dot product rules ---
        case Dot():
            # zero absorption: dot(0, x) or dot(x, 0)
            if expr.left.is_zero or expr.right.is_zero:
                return _ZERO()

            match (expr.left, expr.right):
                # dot(x, cross(a, b)) — cross on right
                case (_, Cross()):
                    if _is_orthogonal_dot_cross(expr.left, expr.right):
                        return _ZERO()
                    return expr

                # dot(cross(a, b), x) — cross on left
                case (Cross(), _):
                    if _is_orthogonal_dot_cross(expr.right, expr.left):
                        return _ZERO()
                    return expr

                # dot(q, q) = 1 when unit norm
                case _ if (
                    expr.left.is_unit_norm and expr.right.is_unit_norm and expr.left == expr.right
                ):
                    return _ONE()

                # dot(ω, q) = 0 — tangent vector orthogonal to manifold point
                case _ if _is_tangent_orthogonal(expr.left, expr.right):
                    return _ZERO()

                case _:
                    return expr

        # --- Cross product rules ---
        case Cross() if expr.left == expr.right:
            return ZeroVector

        case Cross() if expr.left.is_zero or expr.right.is_zero:
            return ZeroVector

        # cross(a, s*a) = s*(a × a) = 0  and  cross(s*a, a) = 0
        case Cross() if isinstance(expr.right, SVMul) and expr.right.left == expr.left:
            return ZeroVector

        case Cross() if isinstance(expr.left, SVMul) and expr.left.left == expr.right:
            return ZeroVector

        # S2 BAC-CAB identity with tangent orthogonality.
        # First recurse into children so inner reductions fire before outer.
        case Cross():
            expr = Cross(vector_rules(expr.left), vector_rules(expr.right))
            reduced = _try_s2_cross_reduction(expr)
            if reduced is not None:
                return reduced
            return expr

        # --- Transpose rules ---
        case Transpose() if isinstance(expr.expr, Transpose):
            return expr.expr.expr

        # --- Hat rules ---
        case Hat() if expr.expr.is_zero:
            return ZeroMatrix

        case _:
            return expr
