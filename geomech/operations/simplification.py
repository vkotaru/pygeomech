from geomech.operations.addition import Add, MAdd, VAdd
from geomech.operations.binary_tree import has_nested_scalars
from geomech.operations.multiplication import Mul, SVMul, MVMul, SMMul, MMMul
from geomech.operations.geometry import Dot, Cross
from geomech.base.matrices import MatrixExpr
from geomech.base.scalars import ScalarExpr, Scalar
from geomech.base.vectors import VectorExpr
from geomech.operations.expansion import expand


def pull(expr):
    """
    Function to pull scalar components out of vectors and matrices
    """
    if isinstance(expr, ScalarExpr):
        if isinstance(expr, Add):
            pulled_expr = Add()
            for n in expr.nodes:
                pulled_expr += pull(n)
            return pulled_expr
        elif isinstance(expr, Mul):
            return pull(expr.left) * pull(expr.right)
        elif isinstance(expr, Dot):
            if isinstance(expr.left, SVMul):
                return pull(expr.left.right * Dot(expr.left.left, expr.right))
            elif isinstance(expr.right, SVMul):
                return pull(expr.right.right * Dot(expr.right.left, expr.left))
            else:
                if has_nested_scalars(expr):
                    return pull(Dot(pull(expr.left), pull(expr.right)))
                else:
                    return expr
        else:
            return expr

    elif isinstance(expr, VectorExpr):
        if isinstance(expr, MVMul):
            if isinstance(expr.left, SMMul):
                return pull(SVMul(MVMul(expr.left.left, expr.right), expr.left.right))
            elif isinstance(expr.right, SVMul):
                return pull(SVMul(MVMul(expr.left, expr.right.left), expr.right.right))
            else:
                if has_nested_scalars(expr):
                    return pull(MVMul(pull(expr.left), pull(expr.right)))
                else:
                    return expr
        elif isinstance(expr, SVMul):
            if isinstance(expr.left, SVMul):
                return pull(SVMul(expr.left.left, expr.left.right * expr.right))
            else:
                if has_nested_scalars(expr):
                    return pull(SVMul(pull(expr.left), pull(expr.right)))
                else:
                    return expr
        elif isinstance(expr, Cross):
            if isinstance(expr.left, SVMul) and isinstance(expr.right, SVMul):
                return pull(SVMul(Cross(expr.left.left, expr.right.left), expr.left.right * expr.right.right))
            elif isinstance(expr.left, SVMul):
                return pull(SVMul(Cross(expr.left.left, expr.right), expr.left.right))
            elif isinstance(expr.right, SVMul):
                return pull(SVMul(Cross(expr.left, expr.right.left), expr.right.right))
            else:
                if has_nested_scalars(expr):
                    return pull(Cross(pull(expr.left), pull(expr.right)))
                else:
                    return expr
        else:
            return expr

    elif isinstance(expr, MatrixExpr):
        if isinstance(expr, MAdd):
            return pull(expr.left) + pull(expr.right)

        elif isinstance(expr, MMMul):
            if isinstance(expr.left, SMMul) and isinstance(expr.right, SMMul):
                return pull(SMMul((expr.left.left * expr.right.left), (expr.left.right * expr.right.right)))
            elif isinstance(expr.right, SMMul):
                return pull(SMMul(MMMul(expr.left, expr.right.left), expr.right.right))
            elif isinstance(expr.left, SMMul):
                return pull(SMMul(MMMul(expr.left.left, expr.right), expr.left.right))
            else:
                if has_nested_scalars(expr):
                    return pull(MMMul(pull(expr.left), pull(expr.right)))
                else:
                    return expr

        elif isinstance(expr, SMMul):
            if isinstance(expr.left, SMMul):
                return pull(SMMul(expr.left.left, expr.left.right * expr.right))
            else:
                if has_nested_scalars(expr):
                    return pull(SMMul(pull(expr.left), expr.right))
                else:
                    return expr
        else:
            return expr

    else:
        return expr


def vector_rules(expr):
    if isinstance(expr, Add):
        ruled_expr = Add()
        for n in expr.nodes:
            ruled_expr += vector_rules(n)
        return ruled_expr
    elif isinstance(expr, Mul):
        return vector_rules(expr.left) * vector_rules(expr.right)

    # gather numerics
    elif isinstance(expr, Dot):
        if isinstance(expr.left, Cross):
            raise NotImplementedError
        elif isinstance(expr.right, Cross):
            if (expr.left == expr.right.left) or (expr.left == expr.right.right) or (
                    expr.right.left == expr.right.right):
                return Scalar('0', value=0, attr=['Constant', 'Zero'])
            else:
                return expr
        else:
            if expr.left.isUnitNorm and expr.right.isUnitNorm and (expr.left == expr.right):
                return Scalar('1', value=1, attr=['Constant', 'Ones'])
            else:
                return expr
    return expr


def combine(expr):
    if isinstance(expr, Mul):
        return combine(expr.left) * combine(expr.right)
    elif expr.value is not None:
        return expr.value
    else:
        return 1


def has_zeros(expr):
    if isinstance(expr, Mul):
        return has_zeros(expr.left) or has_zeros(expr.right)
    elif expr.value is not None:
        if expr.value == 0:
            return True
        else:
            return False
    else:
        return False


def any_terms(expr):
    if isinstance(expr, Mul):
        return any_terms(expr.left) or any_terms(expr.right)
    elif expr.value is not None:
        return False
    else:
        return True


def terms(expr):
    if isinstance(expr, Mul):
        if expr.left.value is not None:
            return terms(expr.right)
        elif expr.right.value is not None:
            return terms(expr.left)
        else:
            if any_terms(expr.left) and any_terms(expr.right):
                return terms(expr.left) * terms(expr.right)
            elif any_terms(expr.left):
                return terms(expr.left)
            else:
                return terms(expr.right)
    else:
        return expr


def simplify(expr):
    """combines constants and eliminates zeros """
    if isinstance(expr, ScalarExpr):  # TODO modify this to Expression.SCALAR?
        if isinstance(expr, Add):
            """Remove zeros"""
            value = 0
            sym_exprs = []
            for n in expr.nodes:
                if n.isZero:
                    continue
                elif n.value is not None:
                    value += n.value
                else:
                    sym_exprs.append(simplify(n))
            if value != 0:
                sym_exprs.append(Scalar(str(value), value=value, attr=['Constant', 'Zero']))
            return Add(sym_exprs)

        elif isinstance(expr, Mul):
            if has_zeros(expr):
                return Scalar('0', value=0, attr=['Constant', 'Zero'])
            elif expr.left.value == 1:
                return expr.right
            elif expr.right.value == 1:
                return expr.left
            else:
                val = combine(expr)
                left = Scalar('(' + str(val) + ')', value=val, attr=['Constant'])
                right = terms(expr)
                if val == 1:
                    return right
                return Mul(left, right)

        elif isinstance(expr, Dot):
            return expr
        else:
            return expr

    elif isinstance(expr, VectorExpr):
        if isinstance(expr, VAdd):
            sym_exprs = []
            for n in expr.nodes:
                if n.isZero:
                    continue
                elif isinstance(n, SVMul) and n.value == 0:
                    continue
                else:
                    sym_exprs.append(simplify(n))
            return VAdd(sym_exprs)
        elif isinstance(expr, SVMul):
            return SVMul(simplify(expr.left), simplify(expr.right))
        else:
            return expr

    elif isinstance(expr, MatrixExpr):
        return

    return expr


def full_simplify(expr):
    """simplifies expr using standard operations (# TODO)"""

    # expand
    expr_ = expand(expr)
    # pull
    expr_ = pull(expr_)
    # apply vector rules
    expr_ = vector_rules(expr_)
    # simplify
    expr_ = simplify(expr_)

    return expr_
