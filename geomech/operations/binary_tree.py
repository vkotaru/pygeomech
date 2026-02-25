from geomech.operations.addition import Add, VAdd, MAdd
from geomech.operations.multiplication import SVMul, SMMul
from geomech.base.matrices import MatrixExpr
from geomech.base.scalars import ScalarExpr
from geomech.base.vectors import VectorExpr
from geomech.base.nodes import BinaryNode


def is_leaf(expr):
    if isinstance(expr, BinaryNode):
        return False
    else:
        return True


def is_member(expr, member):
    if isinstance(expr, BinaryNode):
        return is_member(expr.left, member) and is_member(expr.right, member)  # TODO
    else:
        return expr == member


def has_nested_add(expr):
    if isinstance(expr, BinaryNode):
        if isinstance(expr, Add) or isinstance(expr, VAdd) or isinstance(expr, MAdd):
            return True
        else:
            return has_nested_add(expr.left) or has_nested_add(expr.right)
    else:
        return False


def has_nested_scalars(expr):
    if isinstance(expr, ScalarExpr):
        if isinstance(expr, BinaryNode):
            return has_nested_scalars(expr.left) or has_nested_scalars(expr.right)
        else:
            return False  # TODO what about just scalar addition or multiplication

    elif isinstance(expr, VectorExpr):
        if isinstance(expr, BinaryNode):
            if isinstance(expr, SVMul):
                return True
            else:
                return has_nested_scalars(expr.left) or has_nested_scalars(expr.right)
        else:
            return False

    elif isinstance(expr, MatrixExpr):
        if isinstance(expr, BinaryNode):
            if isinstance(expr, SMMul):
                return True
            else:
                return has_nested_scalars(expr.left) or has_nested_scalars(expr.right)
        else:
            return False
    else:
        return False
