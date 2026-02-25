from geomech import Add, MAdd
from geomech.base.matrices import Matrix, MatrixExpr
from geomech.base.expr import Expr, Expression
from geomech.base.scalars import ScalarExpr, Scalar
from geomech.base.vectors import VectorExpr, Vector
from geomech.base.nodes import BinaryNode
from geomech.operations.transpose import Transpose
from geomech.utils.errors import ExpressionMismatchError, UndefinedCaseError


class Mul(BinaryNode, ScalarExpr):
    """
    Scalar multiplication
    """

    def __init__(self, l, r):
        if type(l) == float or type(l) == int:
            l = Scalar('(' + str(l) + ')', value=l, attr=['Constant'])
        if type(r) == float or type(r) == int:
            r = Scalar('(' + str(r) + ')', value=r, attr=['Constant'])
        if l.type == Expression.SCALAR and r.type == Expression.SCALAR:
            super().__init__(l, r)
            self.type = l.type
        else:
            raise Exception('Input to ', type(self).__name__, 'should be Expression', str(self.type))
        self.isConstant = self.left.isConstant and self.right.isConstant
        self.isZero = self.left.isZero or self.right.isZero

    def __str__(self):
        str = self.left.__str__() + self.right.__str__()
        return str

    def delta(self):
        if self.left.isConstant and not self.right.isConstant:
            return Mul(self.left, self.right.delta())
        elif not self.left.isConstant and self.right.isConstant:
            return Mul(self.left.delta(), self.right)
        elif self.left.isConstant and self.right.isConstant:
            return Scalar('0', attr=['Constant', 'Zero'])
        else:
            return Mul(self.left.delta(), self.right) + Mul(self.left, self.right.delta())

    def diff(self):
        if self.left.isConstant and not self.right.isConstant:
            return Mul(self.left, self.right.diff())
        elif not self.left.isConstant and self.right.isConstant:
            return Mul(self.left.diff(), self.right)
        elif self.left.isConstant and self.right.isConstant:
            return Scalar('0', attr=['Constant', 'Zero'])
        else:
            return Mul(self.left.diff(), self.right) + Mul(self.left, self.right.diff())


class MVMul(BinaryNode, VectorExpr):
    """
    Matrix-Vector multiplication
    """

    def __init__(self, l, r):
        super().__init__()
        if l.type == Expression.MATRIX and r.type == Expression.VECTOR:
            self.left = l
            self.right = r
            self.type = Expression.VECTOR
        elif l.type == Expression.VECTOR and r.type == Expression.MATRIX:
            if type(l) == type(Transpose()):
                self.left = l
                self.right = r
                self.type = Expression.VECTOR
            else:
                raise ExpressionMismatchError
        else:
            raise ExpressionMismatchError
        self.isConstant = self.left.isConstant and self.right.isConstant
        self.isZero = self.left.isZero or self.right.isZero

    def __str__(self):
        str = self.left.__str__() + self.right.__str__()
        return str

    def __mul__(self, other):
        if other.type == Expression.SCALAR:
            return SVMul(self, other)
        elif other.type == Expression.VECTOR:
            return VVMul(self, other)
        elif other.type == Expression.MATRIX:
            return MVMul(self, other)

    def delta(self):
        if self.left.isConstant and not self.right.isConstant:
            return MVMul(self.left, self.right.delta())
        elif not self.left.isConstant and self.right.isConstant:
            return MVMul(self.left.delta(), self.right)
        elif self.left.isConstant and self.right.isConstant:
            return Vector('0', attr=['Constant', 'Zero'])
        else:
            return MVMul(self.left.delta(), self.right) + MVMul(self.left, self.right.delta())

    def diff(self):
        if self.left.isConstant and not self.right.isConstant:
            return MVMul(self.left, self.right.diff())
        elif not self.left.isConstant and self.right.isConstant:
            return MVMul(self.left.diff(), self.right)
        elif self.left.isConstant and self.right.isConstant:
            return Vector('0', attr=['Constant', 'Zero'])
        else:
            return MVMul(self.left.diff(), self.right) + MVMul(self.left, self.right.diff())


class MMMul(BinaryNode, MatrixExpr):
    """
    Matrix-Matrix multiplication
    """

    def __init__(self, l, r):
        super().__init__()
        if l.type == Expression.MATRIX and r.type == Expression.MATRIX:
            self.left = l
            self.right = r
            self.type = Expression.MATRIX
        else:
            raise ExpressionMismatchError
        self.isConstant = self.left.isConstant and self.right.isConstant
        self.isZero = self.left.isZero or self.right.isZero

    def __str__(self):
        str = self.left.__str__() + self.right.__str__()
        return str

    def __mul__(self, other):
        if other.type == Expression.SCALAR:
            return SMMul(self, other)
        elif other.type == Expression.VECTOR:
            return MVMul(self, other)
        elif other.type == Expression.MATRIX:
            return MMMul(self, other)

    def delta(self):
        if self.left.isConstant and not self.right.isConstant:
            return MMMul(self.left, self.right.delta())
        elif not self.left.isConstant and self.right.isConstant:
            return MMMul(self.left.delta(), self.right)
        elif self.left.isConstant and self.right.isConstant:
            return Matrix('O', attr=['Constant', 'Zero'])
        else:
            return MMMul(self.left.delta(), self.right) + MMMul(self.left, self.right.delta())

    def diff(self):
        if self.left.isConstant and not self.right.isConstant:
            return MMMul(self.left, self.right.diff())
        elif not self.left.isConstant and self.right.isConstant:
            return MMMul(self.left.diff(), self.right)
        elif self.left.isConstant and self.right.isConstant:
            return Matrix('O', attr=['Constant', 'Zero'])
        else:
            return MMMul(self.left.diff(), self.right) + MMMul(self.left, self.right.diff())


class SVMul(BinaryNode, VectorExpr):
    """
    Scalar-Vector multiplication
    """

    def __init__(self, l, r):
        super().__init__()
        if type(l) == float or type(l) == int:
            l = Scalar('(' + str(l) + ')', value=l, attr=['Constant'])
        if type(r) == float or type(r) == int:
            r = Scalar('(' + str(r) + ')', value=r, attr=['Constant'])
        if l.type == Expression.VECTOR and r.type == Expression.SCALAR:
            self.left = l
            self.right = r
            self.type = Expression.VECTOR
        elif l.type == Expression.SCALAR and r.type == Expression.VECTOR:
            self.left = r
            self.right = l
            self.type = Expression.VECTOR
        else:
            raise ExpressionMismatchError
        self.isConstant = self.left.isConstant and self.right.isConstant
        self.isZero = self.left.isZero or self.right.isZero

    def __str__(self):
        str = self.left.__str__() + self.right.__str__()
        return str

    def __mul__(self, other):
        if __name__ == '__main__':
            if other.type == Expression.SCALAR:
                return SVMul(self, other)
            elif other.type == Expression.VECTOR:
                return VVMul(self, other)
            elif other.type == Expression.MATRIX:
                return MVMul(self, other)

    def delta(self):
        if self.left.isConstant and not self.right.isConstant:
            return SVMul(self.left, self.right.delta())
        elif not self.left.isConstant and self.right.isConstant:
            return SVMul(self.left.delta(), self.right)
        elif self.left.isConstant and self.right.isConstant:
            return Vector('0', attr=['Constant', 'Zero'])
        else:
            return SVMul(self.left.delta(), self.right) + SVMul(self.left, self.right.delta())

    def diff(self):
        if self.left.isConstant and not self.right.isConstant:
            return SVMul(self.left, self.right.diff())
        elif not self.left.isConstant and self.right.isConstant:
            return SVMul(self.left.diff(), self.right)
        elif self.left.isConstant and self.right.isConstant:
            return Vector('0', attr=['Constant', 'Zero'])
        else:
            return SVMul(self.left.diff(), self.right) + SVMul(self.left, self.right.diff())


class SMMul(BinaryNode, MatrixExpr):
    """
    Scalar-Matrix Multiplication
    """

    def __init__(self, l, r):
        super().__init__()
        if type(l) == float or type(l) == int:
            l = Scalar('(' + str(l) + ')', value=l, attr=['Constant'])
        if type(r) == float or type(r) == int:
            r = Scalar('(' + str(r) + ')', value=r, attr=['Constant'])

        if l.type == Expression.MATRIX and r.type == Expression.SCALAR:
            self.left = l
            self.right = r
            self.type = Expression.MATRIX
        elif l.type == Expression.SCALAR and r.type == Expression.MATRIX:
            self.left = r
            self.right = l
            self.type = Expression.MATRIX
        else:
            raise ExpressionMismatchError
        self.isConstant = self.left.isConstant and self.right.isConstant
        self.isZero = self.left.isZero or self.right.isZero

    def __str__(self):
        str = self.left.__str__() + self.right.__str__()
        return str

    def __mul__(self, other):
        if other.type == Expression.SCALAR:
            return SMMul(self, other)
        elif other.type == Expression.VECTOR:
            return MVMul(self, other)
        elif other.type == Expression.MATRIX:
            return MMMul(self, other)
        else:
            raise UndefinedCaseError

    def delta(self):
        if self.left.isConstant and not self.right.isConstant:
            return SMMul(self.left, self.right.delta())
        elif not self.left.isConstant and self.right.isConstant:
            return SMMul(self.left.delta(), self.right)
        elif self.left.isConstant and self.right.isConstant:
            return Matrix('O', attr=['Constant', 'Zero'])
        else:
            return SMMul(self.left.delta(), self.right) + SMMul(self.left, self.right.delta())

    def diff(self):
        if self.left.isConstant and not self.right.isConstant:
            return SMMul(self.left, self.right.diff())
        elif not self.left.isConstant and self.right.isConstant:
            return SMMul(self.left.diff(), self.right)
        elif self.left.isConstant and self.right.isConstant:
            return Matrix('O', attr=['Constant', 'Zero'])
        else:
            return SMMul(self.left.diff(), self.right) + SMMul(self.left, self.right.diff())


class VVMul(Expr, BinaryNode):
    """
    Vector Vector Multiplication
    """

    def __init__(self, l, r):
        super().__init__()
        if l.type == Expression.VECTOR and r.type == Expression.VECTOR:
            if type(l) == type(Transpose(None)) and type(r) != type(Transpose(None)):
                self.left = l
                self.right = r
                self.type = Expression.SCALAR
            elif type(l) != type(Transpose(None)) and type(r) == type(Transpose(None)):
                self.left = l
                self.right = r
                self.type = Expression.MATRIX
            else:
                raise ExpressionMismatchError
        else:
            raise ExpressionMismatchError
        self.isConstant = self.left.isConstant and self.right.isConstant
        self.isZero = self.left.isZero or self.right.isZero

    def __str__(self):
        return self.left.__str__() + self.right.__str__()

    def __add__(self, other):
        if self.type == Expression.SCALAR:
            return Add(self, other)
        elif self.type == Expression.MATRIX:
            return MAdd(self, other)
        else:
            raise UndefinedCaseError

    def __mul__(self, other):
        if self.type == Expression.SCALAR:
            if other.type == Expression.SCALAR:
                return Mul(self, other)
            elif other.type == Expression.VECTOR:
                return SVMul(self, other)
            elif other.type == Expression.MATRIX:
                return SMMul(self, other)
            else:
                raise UndefinedCaseError
        elif self.type == Expression.MATRIX:
            if other.type == Expression.SCALAR:
                return SMMul(self, other)
            elif other.type == Expression.VECTOR:
                return MVMul(self, other)
            elif other.type == Expression.MATRIX:
                return MMMul(self, other)
            else:
                raise UndefinedCaseError
        else:
            raise UndefinedCaseError

    def delta(self):
        if self.left.isConstant and not self.right.isConstant:
            return VVMul(self.left, self.right.delta())
        elif not self.left.isConstant and self.right.isConstant:
            return VVMul(self.left.delta(), self.right)
        elif self.left.isConstant and self.right.isConstant:
            if self.type == Expression.SCALAR:
                return Scalar('0', attr=['Constant', 'Zero'])
            elif self.type == Expression.MATRIX:
                return Matrix('O', attr=['Constant', 'Zero'])
            else:
                raise UndefinedCaseError
        else:
            return VVMul(self.left.delta(), self.right) + VVMul(self.left, self.right.delta())

    def diff(self):
        if self.left.isConstant and not self.right.isConstant:
            return VVMul(self.left, self.right.diff())
        elif not self.left.isConstant and self.right.isConstant:
            return VVMul(self.left.diff(), self.right)
        elif self.left.isConstant and self.right.isConstant:
            if self.type == Expression.SCALAR:
                return Scalar('0', attr=['Constant', 'Zero'])
            elif self.type == Expression.MATRIX:
                return Matrix('O', attr=['Constant', 'Zero'])
            else:
                raise UndefinedCaseError
        else:
            return VVMul(self.left.diff(), self.right) + VVMul(self.left, self.right.diff())
