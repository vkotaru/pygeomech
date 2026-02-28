"""Pull scalar factors out of vector/matrix operations.

Bottom-up traversal: recurse into children first, then handle the
current node.  O(n) single pass — no has_nested_scalars predicate needed.
"""
from geomech.core.operations.addition import Add, VAdd, MAdd
from geomech.core.operations.multiplication import Mul, SVMul, SMMul, MVMul, MMMul
from geomech.core.operations.geometry import Dot, Cross
from geomech.core.operations.calculus import Variation, TimeDerivative, TimeIntegral


def pull(expr):
    match expr:
        # --- N-ary (additions) ---
        case Add(nodes=nodes):
            return Add(*[pull(n) for n in nodes])
        case VAdd(nodes=nodes):
            return VAdd(*[pull(n) for n in nodes])
        case MAdd(nodes=nodes):
            return MAdd(*[pull(n) for n in nodes])

        # --- Scalar * Scalar ---
        case Mul():
            return Mul(pull(expr.left), pull(expr.right))

        # --- Dot(vec, vec) → scalar ---
        case Dot():
            l, r = pull(expr.left), pull(expr.right)
            match (l, r):
                case (SVMul(), SVMul()):
                    return Mul(l.right, Mul(r.right, Dot(l.left, r.left)))
                case (SVMul(), _):
                    return Mul(l.right, Dot(l.left, r))
                case (_, SVMul()):
                    return Mul(r.right, Dot(l, r.left))
                case _:
                    return Dot(l, r)

        # --- Scalar * Vector ---
        case SVMul():
            l, r = pull(expr.left), pull(expr.right)
            match l:
                case SVMul():
                    return SVMul(l.left, Mul(l.right, r))
                case _:
                    return SVMul(l, r)

        # --- Cross(vec, vec) → vector ---
        case Cross():
            l, r = pull(expr.left), pull(expr.right)
            match (l, r):
                case (SVMul(), SVMul()):
                    return SVMul(Cross(l.left, r.left), Mul(l.right, r.right))
                case (SVMul(), _):
                    return SVMul(Cross(l.left, r), l.right)
                case (_, SVMul()):
                    return SVMul(Cross(l, r.left), r.right)
                case _:
                    return Cross(l, r)

        # --- Matrix * Vector → vector ---
        case MVMul():
            l, r = pull(expr.left), pull(expr.right)
            match (l, r):
                case (SMMul(), SVMul()):
                    return SVMul(MVMul(l.left, r.left), Mul(l.right, r.right))
                case (SMMul(), _):
                    return SVMul(MVMul(l.left, r), l.right)
                case (_, SVMul()):
                    return SVMul(MVMul(l, r.left), r.right)
                case _:
                    return MVMul(l, r)

        # --- Scalar * Matrix ---
        case SMMul():
            l, r = pull(expr.left), pull(expr.right)
            match l:
                case SMMul():
                    return SMMul(l.left, Mul(l.right, r))
                case _:
                    return SMMul(l, r)

        # --- Matrix * Matrix → matrix ---
        case MMMul():
            l, r = pull(expr.left), pull(expr.right)
            match (l, r):
                case (SMMul(), SMMul()):
                    return SMMul(MMMul(l.left, r.left), Mul(l.right, r.right))
                case (SMMul(), _):
                    return SMMul(MMMul(l.left, r), l.right)
                case (_, SMMul()):
                    return SMMul(MMMul(l, r.left), r.right)
                case _:
                    return MMMul(l, r)

        # --- Calculus unary: recurse into child ---
        case Variation():
            return Variation(pull(expr.expr))

        case TimeDerivative():
            return TimeDerivative(pull(expr.expr))

        case TimeIntegral():
            return TimeIntegral(pull(expr.expr))

        # --- Leaf / unhandled ---
        case _:
            return expr
