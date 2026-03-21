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
from geomech.core.base.expressions import Scalar, ZeroVector, ZeroMatrix, TS2, S2
from geomech.core.operations.addition import Add, VAdd
from geomech.core.operations.multiplication import Mul, SVMul
from geomech.core.operations.geometry import Dot, Cross, Hat, Transpose


def _is_orthogonal_dot_cross(vec, cross_expr):
    """Check if dot(vec, cross(a, b)) is zero by orthogonality.

    Returns True when:
      - vec == a or vec == b  (x perpendicular to x x y)
      - a == b                (x x x = 0)
    """
    return (vec == cross_expr.left
            or vec == cross_expr.right
            or cross_expr.left == cross_expr.right)


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
    """Try to simplify Cross expressions using S2 tangent space identities.

    BAC-CAB: q × (a × q) = a*(q·q) - q*(q·a)
    When q is unit norm (S2) and a is in the tangent space (TS2 with parent q):
      q × (a × q) = a

    Also handles the flipped form:
      cross(cross(a, q), q) = -cross(q, cross(a, q)) ... but
      cross(cross(q, a), q) = a  (by anti-commutativity of inner cross)
    """
    l, r = expr.left, expr.right

    # Pattern: Cross(q, Cross(a, q)) where q is S2, a is tangent to q
    if (isinstance(l, S2) and isinstance(r, Cross)
            and r.right == l):
        a = r.left
        if _is_tangent_to_s2(a, l):
            return a

    # Pattern: Cross(q, Cross(q, a)) = -a (when a is tangent)
    # q × (q × a) = q*(q·a) - a*(q·q) = -a  (since q·a=0 and q·q=1)
    if (isinstance(l, S2) and isinstance(r, Cross)
            and r.left == l):
        a = r.right
        if _is_tangent_to_s2(a, l):
            return SVMul(a, Scalar('(-1)', value=-1, attr=['Constant']))

    # Pattern: Cross(Cross(q, a), q) = a (when a is tangent)
    # (q × a) × q = q*(a·q) - a*(q·q) ... no, BAC-CAB is a×(b×c)
    # Actually: (q×a) × q = -q × (q×a) = -(q*(q·a) - a*(q·q)) = a
    if (isinstance(l, Cross) and isinstance(r, S2)
            and l.left == r):
        a = l.right
        if _is_tangent_to_s2(a, r):
            return a

    # Pattern: Cross(Cross(a, q), q) = -a (when a is tangent)
    # (a×q) × q = -q × (a×q) = -(a*(q·q) - q*(q·a)) = -a
    if (isinstance(l, Cross) and isinstance(r, S2)
            and l.right == r):
        a = l.left
        if _is_tangent_to_s2(a, r):
            return SVMul(a, Scalar('(-1)', value=-1, attr=['Constant']))

    # Pattern: Cross(a, Cross(a, q)) = -q*(a·a) when a is tangent to q
    # a × (a × q) = a*(a·q) - q*(a·a) = -q*(a·a) since a⊥q
    # This produces SVMul(q, -Dot(a,a)) = scalar * q
    if (isinstance(r, Cross) and r.right is not None
            and isinstance(r.right, S2) and l == r.left):
        q = r.right
        a = l
        if _is_tangent_to_s2(a, q):
            neg_one = Scalar('(-1)', value=-1, attr=['Constant'])
            return SVMul(q, Mul(Dot(a, a), neg_one))

    return None


def _is_tangent_to_s2(expr, s2):
    """Check if expr is a tangent vector to the given S2 manifold.

    Returns True for TS2 vectors whose parent is the given S2,
    or for TimeDerivative/Variation of such vectors.
    """
    from geomech.core.operations.calculus import TimeDerivative, Variation
    # Direct TS2
    if isinstance(expr, TS2) and expr.S2 is s2:
        return True
    # TimeDerivative(TS2) — angular acceleration stays in tangent space
    if isinstance(expr, TimeDerivative):
        return _is_tangent_to_s2(expr.expr, s2)
    # Variation(TS2)
    if isinstance(expr, Variation):
        return _is_tangent_to_s2(expr.expr, s2)
    # Cross(a, q) is tangent to q (perpendicular to q by definition)
    if isinstance(expr, Cross):
        if expr.right == s2 or expr.left == s2:
            return True
    return False


_ZERO = lambda: Scalar('0', value=0, attr=['Constant', 'Zero'])
_ONE = lambda: Scalar('1', value=1, attr=['Constant', 'Ones'])


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
                case _ if (expr.left.is_unit_norm
                           and expr.right.is_unit_norm
                           and expr.left == expr.right):
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

        # S2 tangent recovery: cross(q, cross(a, q)) = a  when q unit norm, a ⊥ q
        # This is the BAC-CAB identity: q × (a × q) = a(q·q) - q(q·a) = a when ||q||=1 and a⊥q
        # First recurse into children so inner reductions fire before outer
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
