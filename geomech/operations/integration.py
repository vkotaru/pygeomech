from geomech.base.expr import Expression
from geomech.operations.collection import col
from geomech.operations.expansion import expand
from geomech.operations.simplification import full_simplify
from geomech.operations.geometry import Dot
from geomech.operations.addition import Add
from geomech.operations.multiplication import Mul


def ibp(_expr, _expr_int):
    """implements integration by parts on a collection term
    :_expr_int: expression to be integrated
    :_expr
    """
    if _expr_int.type == Expression.VECTOR:
        # parentheses not implemented TODO
        if isinstance(_expr, Mul):
            return Mul(ibp(_expr.left, _expr_int), ibp(_expr.right, _expr_int))
        elif isinstance(_expr, Add):
            ibp_expr = Add()
            for n in _expr.nodes:
                ibp_expr += ibp(n, _expr_int)
            return ibp_expr
        elif isinstance(_expr, Dot):
            if _expr.left == _expr_int:
                return Dot(_expr.left.integrate() * (-1), _expr.right.diff())
            else:
                return _expr
        else:
            return _expr

    #   case Par(u)   => Par(ibp(u, old))
    #   case Mul(u,v) => {Mul(ibp(u, old),ibp(v, old))}
    #   case Add(u,v) => {Add(ibp(u, old),ibp(v, old))}
    #   case Dot(u,v) => if (old == u) {Dot(integrate(u)*Num(-1),diffV(v))} else {Dot(u,v)}
    #   case u:Exp => u

    elif _expr_int.type == Expression.SCALAR:
        raise NotImplementedError
    #   case Add(u,v) => ibpScalar(u,old) + ibpScalar(v,old)
    #   case Mul(u,v) => if (u == old) {Mul(Num(-1),integrateScalar(u))*diffS(v)} else {u*v}
    #   case u:Exp => u
    return _expr


def integrate_by_parts_vectors(expr, vectors):
    # TODO update and clean this function
    expr = expand(expr)
    expr = full_simplify(expr)
    for vector in vectors:
        expr = col(expr, vector)  # TODO clean and update this function
        expr = ibp(expr, vector)
        expr = full_simplify(expr)

    return expr


def integrate_by_parts_scalars(expr, scalars):
    expr = expand(expr)
    expr = full_simplify(expr)
