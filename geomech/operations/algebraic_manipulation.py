from geomech.operations.transpose import Transpose
from geomech.base.matrices import Matrix, ZeroMatrix
from geomech.base.vectors import Vector
from geomech.base.expr import Expression
from geomech.utils.errors import UndefinedCaseError
from geomech.operations.addition import Add, VAdd, MAdd
from geomech.operations.multiplication import Mul, SVMul, SMMul, VVMul, MVMul, MMMul
from geomech.operations.geometry import Dot, Cross, Hat


def extract_coeff(expr, sym):
    if expr.type == Expression.SCALAR:
        return efs(expr, sym)

    elif expr.type == Expression.VECTOR:
        raise NotImplementedError

    elif expr.type == Expression.MATRIX:
        raise NotImplementedError
    else:
        raise UndefinedCaseError


def efs(expr, vec):
    """Extract From Scalar"""
    # Add
    if isinstance(expr, Add):
        sym_efs_exprs = []
        for n in expr.nodes:
            if n.has(vec):
                sym_efs_exprs.append(efs(n, vec))
        return VAdd(sym_efs_exprs)
        # if expr.left.has(vec) and expr.right.has(vec):
        #     return VAdd(efs(expr.left, vec), efs(expr.right, vec))
        # elif expr.left.has(vec):
        #     return efs(expr.left, vec)
        # elif expr.right.has(vec):
        #     return efs(expr.right, vec)
        # else:
        #     return Vector('0', attr=['Constant', 'Zero'])

    elif isinstance(expr, Mul):
        if expr.left.has(vec) and expr.right.has(vec):
            raise NotImplementedError
        elif expr.left.has(vec):
            return SVMul(efs(expr.left, vec), expr.right)
        elif expr.right.has(vec):
            return SVMul(efs(expr.right, vec), expr.left)
        else:
            return Vector('0', attr=['Constant', 'Zero'])

    elif isinstance(expr, Dot):
        if expr.left.has(vec) and expr.right.has(vec):
            raise NotImplementedError
        elif expr.left.has(vec):
            if expr.left == vec:
                return expr.right
            else:
                if expr.left.type == Expression.VECTOR:
                    return MVMul(Transpose(efv(expr.left, vec)), expr.right)
                else:
                    raise UndefinedCaseError
                # if isinstance(expr.left, MVMul):
                #     if expr.left.left.has(vec):
                #         if isinstance(expr.left.left, MVMul):
                #             raise NotImplementedError
                #         elif isinstance(expr.left.left, VVMul):
                #             raise NotImplementedError
                #         else:
                #             raise NotImplementedError
                #     elif expr.left.right.has(vec):
                #         return MVMul(Transpose(expr.left.left),expr.right)
                #     else:
                #         raise UndefinedCaseError
        elif expr.right.has(vec):
            if expr.right == vec:
                return expr.left
            else:
                if expr.right.type == Expression.VECTOR:
                    return efv(expr.right, vec) * expr.left
                else:
                    raise UndefinedCaseError
        else:
            return Vector('0', attr=['Constant', 'Zero'])

    elif isinstance(expr, VVMul):
        raise NotImplementedError

    return expr


def efv(expr, vec):
    """extract from vector"""
    # VAdd
    if isinstance(expr, VAdd):
        if expr.left.has(vec) and expr.right.has(vec):
            return MAdd(efv(expr.left, vec), efv(expr.right, vec))
        elif expr.left.has(vec):
            return efv(expr.left, vec)
        elif expr.right.has(vec):
            return efv(expr.right, vec)
        else:
            return ZeroMatrix

    elif isinstance(expr, Cross):
        if expr.left == expr.right:
            return ZeroMatrix
        elif expr.right == vec:
            return Hat(expr.left)
        elif expr.left == vec:
            return SMMul(Hat(expr.right), -1)
        else:
            if expr.left.has(vec) and expr.right.has(vec):
                raise UndefinedCaseError
            elif expr.left.has(vec):
                return MMMul(SMMul(Hat(expr.right), -1), efv(expr.left, vec))
            elif expr.right.has(vec):
                MMMul(Hat(expr.left), efv(expr.right, vec))
            else:
                return ZeroMatrix

    elif isinstance(expr, MVMul):
        if expr.right.has(vec):
            if expr.right == vec:
                return expr.left
            else:
                return MMMul(expr.left, efv(expr.right, vec))
        elif expr.left.has(vec):
            A, b = expr.left, expr.right
            if isinstance(A, MAdd):
                if A.left.has(vec) and A.right.has(vec):
                    return efv(MVMul(A.left, b), vec) + efv(MVMul(A.right, b), vec)
                elif A.left.has(vec):
                    return efv(MVMul(A.left, b), vec)
                elif A.right.has(vec):
                    return efv(MVMul(A.right, b), vec)
                else:
                    return ZeroMatrix
            elif isinstance(A, MMMul):
                if A.left.has(vec):
                    return efv(MVMul(A.left, MVMul(A.right, b)), vec)
                elif A.right.has(vec):
                    return MMMul(A.left, efv(MVMul(A.right, b), vec))
                else:
                    return ZeroMatrix
            elif isinstance(A, SMMul):
                M, s = A.left, A.right
                if M.has(vec):
                    return efv(MVMul(M, SVMul(b, s)), vec)
                elif s.has(vec):
                    return MVMul(M * b, efs(s, vec))
                else:
                    return ZeroMatrix
            elif isinstance(A, VVMul):
                raise NotImplementedError

            elif isinstance(A, Hat):
                if A.expr == vec:
                    return SMMul(Hat(b), -1)
                else:
                    return MMMul(SMMul(Hat(b), -1), efv(A.expr, vec))

            else:
                return ZeroMatrix
        else:
            return ZeroMatrix

    elif isinstance(expr, SVMul):
        if expr.left.has(vec) and expr.right.has(vec):
            raise UndefinedCaseError

        raise NotImplementedError
    else:
        raise NotImplementedError


def efm(expr, vec):
    """extract from matrix"""
    # MAdd
    if isinstance(expr, MAdd):
        if expr.left.has(vec) and expr.right.has(vec):
            return MAdd(efm(expr.left, vec), efm(expr.right, vec))
        elif expr.left.has(vec):
            return efm(expr.left, vec)
        elif expr.right.has(vec):
            return efm(expr.right, vec)
        else:
            return ZeroMatrix

    elif isinstance(expr, MMMul):
        if expr.left.has(vec) and expr.right.has(vec):
            raise UndefinedCaseError
        elif expr.left.has(vec):
            raise NotImplementedError
        elif expr.right.has(vec):
            if expr.right == vec:
                return expr.left
            else:
                raise NotImplementedError
        else:
            raise UndefinedCaseError
    elif isinstance(expr, VVMul):
        raise NotImplementedError
    elif isinstance(expr, SMMul):
        raise NotImplementedError
    else:
        raise UndefinedCaseError


def solve_for(expr, variable):
    raise NotImplementedError
