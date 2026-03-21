"""Distribute multiplication over addition (expand products).

Bottom-up traversal: expand children first, then distribute at the
current node.  Single pass — no has_nested_add predicate needed.

Distribution rules implemented:
  Mul  over Add   → Add     (scalar)
  Dot  over VAdd  → Add     (scalar from vectors)
  Cross over VAdd → VAdd    (vector)
  SVMul over VAdd/Add → VAdd (vector, left=vector right=scalar)
  MVMul over MAdd/VAdd → VAdd (vector)
  SMMul over MAdd/Add → MAdd (matrix, left=matrix right=scalar)
  MMMul over MAdd → MAdd    (matrix)
  Hat/Vee — recurse into child
"""
from geomech.core.operations.addition import Add, VAdd, MAdd
from geomech.core.operations.multiplication import Mul, SVMul, SMMul, MVMul, MMMul
from geomech.core.operations.geometry import Dot, Cross, Hat, Vee
from geomech.core.operations.calculus import Variation, TimeDerivative, TimeIntegral


def _distribute(l, r, op_cls, sum_cls, l_sum_cls, r_sum_cls):
    """Distribute *op_cls* over addition on either or both sides.

    l, r        — already-expanded left and right children
    op_cls      — constructor for the binary operation (Mul, Dot, …)
    sum_cls     — constructor for the result addition (Add, VAdd, MAdd)
    l_sum_cls   — addition type to check on the left  (VAdd, MAdd, Add)
    r_sum_cls   — addition type to check on the right (VAdd, MAdd, Add)

    If both sides are sums  → cartesian product of terms.
    If one side is a sum    → distribute over that side.
    Otherwise               → reconstruct with expanded children.
    """
    l_is_sum = isinstance(l, l_sum_cls)
    r_is_sum = isinstance(r, r_sum_cls)

    if l_is_sum and r_is_sum:
        return sum_cls(*[op_cls(nl, nr)
                         for nl in l.nodes for nr in r.nodes])
    if l_is_sum:
        return sum_cls(*[op_cls(nl, r) for nl in l.nodes])
    if r_is_sum:
        return sum_cls(*[op_cls(l, nr) for nr in r.nodes])
    return op_cls(l, r)


def expand(expr):
    match expr:
        # --- N-ary additions: expand each term ---
        case Add(nodes=nodes):
            return Add(*[expand(n) for n in nodes])

        case VAdd(nodes=nodes):
            return VAdd(*[expand(n) for n in nodes])

        case MAdd(nodes=nodes):
            return MAdd(*[expand(n) for n in nodes])

        # --- Scalar * Scalar ---
        case Mul():
            l, r = expand(expr.left), expand(expr.right)
            return _distribute(l, r, Mul, Add, Add, Add)

        # --- Dot product (vector x vector → scalar) ---
        case Dot():
            l, r = expand(expr.left), expand(expr.right)
            return _distribute(l, r, Dot, Add, VAdd, VAdd)

        # --- Cross product (vector x vector → vector) ---
        case Cross():
            l, r = expand(expr.left), expand(expr.right)
            return _distribute(l, r, Cross, VAdd, VAdd, VAdd)

        # --- Scalar * Vector  (left=vector, right=scalar) ---
        case SVMul():
            l, r = expand(expr.left), expand(expr.right)
            return _distribute(l, r, SVMul, VAdd, VAdd, Add)

        # --- Matrix * Vector ---
        case MVMul():
            l, r = expand(expr.left), expand(expr.right)
            return _distribute(l, r, MVMul, VAdd, MAdd, VAdd)

        # --- Scalar * Matrix  (left=matrix, right=scalar) ---
        case SMMul():
            l, r = expand(expr.left), expand(expr.right)
            return _distribute(l, r, SMMul, MAdd, MAdd, Add)

        # --- Matrix * Matrix ---
        case MMMul():
            l, r = expand(expr.left), expand(expr.right)
            return _distribute(l, r, MMMul, MAdd, MAdd, MAdd)

        # --- Unary ops: recurse into child ---
        # Hat and Vee are linear: Hat(a + b) = Hat(a) + Hat(b)
        case Hat():
            inner = expand(expr.expr)
            if isinstance(inner, VAdd):
                return MAdd(*[Hat(n) for n in inner.nodes])
            return Hat(inner)

        case Vee():
            inner = expand(expr.expr)
            if isinstance(inner, VAdd):
                return VAdd(*[Vee(n) for n in inner.nodes])
            return Vee(inner)

        case Variation():
            return Variation(expand(expr.expr))

        case TimeDerivative():
            return TimeDerivative(expand(expr.expr))

        case TimeIntegral():
            return TimeIntegral(expand(expr.expr))

        # --- Leaves and everything else ---
        case _:
            return expr
