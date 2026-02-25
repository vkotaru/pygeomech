from geomech.operations.binary_tree import has_nested_add
from geomech.operations.geometry import Dot, Cross, Vee, Hat
from geomech.operations.addition import Add, VAdd, MAdd
from geomech.operations.multiplication import Mul, SMMul, SVMul, MVMul, VVMul, MMMul
from geomech.base.matrices import MatrixExpr
from geomech.base.scalars import ScalarExpr
from geomech.base.vectors import VectorExpr
from geomech.utils.errors import UndefinedCaseError


def expand_scalar(expr):
    if isinstance(expr, Add):
        expanded_expr = Add()
        for n in expr.nodes:
            expanded_expr += expand(n)
        return expanded_expr

    elif isinstance(expr, Mul):
        if isinstance(expr.left, Add) and isinstance(expr.right, Add):
            """(a+b)(c+d) = ac + ad + bc + bd"""
            expanded_expr = Add()
            for nl in expr.left.nodes:
                for nr in expr.right.nodes:
                    expanded_expr += expand(nl * nr)
            return expanded_expr

        elif isinstance(expr.left, Add):
            """(a+b)c = ac + bc"""
            expanded_expr = Add()
            for n in expr.left.nodes:
                expanded_expr += expand(n * expr.right)
            return expanded_expr

        elif isinstance(expr.right, Add):
            """a(b+c) = ab + ac"""
            expanded_expr = Add()
            for n in expr.right.nodes:
                expanded_expr += expand(expr.left * n)
            return expanded_expr

        else:
            if has_nested_add(expr):
                return expand(expand(expr.left) * expand(expr.right))
            else:
                return expr

    elif isinstance(expr, Dot):
        if isinstance(expr.left, VAdd) and isinstance(expr.right, VAdd):
            """(x+y).(u+v) = x.u + x.v + y.u + y.v"""
            expanded_expr = Add()
            for nl in expr.left.nodes:
                for nr in expr.right.nodes:
                    expanded_expr += expand(Dot(nl, nr))
            return expanded_expr
        elif isinstance(expr.right, VAdd):
            """x.(u+v) = x.u + x.v"""
            expanded_expr = Add()
            for n in expr.right.nodes:
                expanded_expr += expand(Dot(expr.left, n))
            return expanded_expr
        elif isinstance(expr.left, VAdd):
            """(x+y).u = x.u + y.u"""
            expanded_expr = Add()
            for n in expr.left.nodes:
                expanded_expr += expand(Dot(n, expr.right))
            return expanded_expr
        else:
            if has_nested_add(expr):
                return expand(Dot(expand(expr.left), expand(expr.right)))
            else:
                return Dot(expand(expr.left), expand(expr.right))

    elif isinstance(expr, VVMul):
        raise NotImplementedError

    return expr


def expand_vector(expr):
    if isinstance(expr, VAdd):
        expanded_expr = VAdd()
        for n in expr.nodes:
            expanded_expr += expand(n)
        return expanded_expr

    elif isinstance(expr, MVMul):
        if isinstance(expr.left, MAdd):
            """(A+B)x = Ax+Bx"""
            expanded_expr = VAdd()
            for n in expr.left.nodes:
                expanded_expr += expand(MVMul(n, expr.right))
            return expanded_expr
        elif isinstance(expr.right, VAdd):
            """A(x+y) = Ax + Ay"""
            expanded_expr = VAdd()
            for n in expr.right.nodes:
                expanded_expr += expand(MVMul(expr.left, n))
            return expanded_expr
        else:
            if has_nested_add(expr):
                return expand(MVMul(expand(expr.left), expand(expr.right)))
            else:
                return expr

    elif isinstance(expr, SVMul):
        if isinstance(expr.left, VAdd):
            """(x+y)a=xa+ya"""
            expanded_expr = VAdd()
            for n in expr.left.nodes:
                expanded_expr += expand(SVMul(n, expr.right))
            return expanded_expr
        else:
            if has_nested_add(expr):
                return expand(SVMul(expand(expr.left), expand(expr.right)))
            else:
                return expr
        pass

    elif isinstance(expr, Cross):
        if isinstance(expr.left, VAdd) and isinstance(expr.right, VAdd):
            expanded_expr = VAdd()
            for nl in expr.left.nodes:
                for nr in expr.right.nodes:
                    expanded_expr += expand(Cross(nl, nr))
            return expanded_expr
        elif isinstance(expr.left, VAdd):
            expanded_expr = VAdd()
            for n in expr.left.nodes:
                expanded_expr += expand(Cross(n, expr.right))
            return expanded_expr
        elif isinstance(expr.right, VAdd):
            """x.(u+v) = x.u + x.v"""
            expanded_expr = VAdd()
            for n in expr.right.nodes:
                expanded_expr += expand(Cross(expr.left, n))
            return expanded_expr
        else:
            if has_nested_add(expr):
                return expand(Cross(expand(expr.left), expand(expr.right)))
            else:
                return expr

    elif isinstance(expr, Vee):
        return Vee(expand(expr))

    return expr


def expand_matrix(expr):
    if isinstance(expr, MAdd):
        expanded_expr = MAdd()
        for n in expr.nodes:
            expanded_expr += expand(n)
        return expanded_expr

    elif isinstance(expr, MMMul):
        if isinstance(expr.left, MAdd) and isinstance(expr.right, MAdd):
            expanded_expr = MAdd()
            for nl in expr.left.nodes:
                for nr in expr.right.nodes:
                    expanded_expr += expand(nl * nr)
            return expanded_expr
        elif isinstance(expr.left, MAdd):
            expanded_expr = MAdd()
            for nl in expr.left.nodes:
                expanded_expr += expand(nl * expr.right)
            return expanded_expr
        elif isinstance(expr.right, MAdd):
            expanded_expr = MAdd()
            for nr in expr.right.nodes:
                expanded_expr += expand(expr.left * nr)
            return expanded_expr
        else:
            if has_nested_add(expr):
                return expand(MMMul(expand(expr.left), expand(expr.right)))
            else:
                return expr

    elif isinstance(expr, SMMul):
        raise Exception('SSMul in expand_matrix is not implemented')

    elif isinstance(expr, VVMul):
        raise Exception('VVMul in expand_matrix is not implemented')

    elif isinstance(expr, Hat):
        return Hat(expand(expr.expr))

    return expr


def expand(expr):
    # TODO add expand functionality to the Expr class directly
    if isinstance(expr, ScalarExpr):
        return expand_scalar(expr)
    elif isinstance(expr, VectorExpr):
        return expand_vector(expr)
    elif isinstance(expr, MatrixExpr):
        return expand_matrix(expr)
    else:
        raise UndefinedCaseError
