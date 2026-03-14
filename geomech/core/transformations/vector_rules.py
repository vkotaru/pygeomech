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
  - Transpose(Transpose(x)) = x  double transpose cancellation
  - Hat(0) = ZeroMatrix           hat of zero vector
  - Linearity through Add and Mul
"""
from geomech.core.base.expressions import Scalar, ZeroVector, ZeroMatrix, TS2, S2
from geomech.core.operations.addition import Add
from geomech.core.operations.multiplication import Mul
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


_ZERO = lambda: Scalar('0', value=0, attr=['Constant', 'Zero'])
_ONE = lambda: Scalar('1', value=1, attr=['Constant', 'Ones'])


def vector_rules(expr):
    match expr:
        # --- Linearity through scalar ops ---
        case Add(nodes=nodes):
            return Add(*[vector_rules(n) for n in nodes])

        case Mul():
            return Mul(vector_rules(expr.left), vector_rules(expr.right))

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

        # --- Transpose rules ---
        case Transpose() if isinstance(expr.expr, Transpose):
            return expr.expr.expr

        # --- Hat rules ---
        case Hat() if expr.expr.is_zero:
            return ZeroMatrix

        case _:
            return expr
