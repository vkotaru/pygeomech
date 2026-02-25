from abc import ABC

from geomech.base.expr import Expr, Expression
from geomech.utils.errors import ExpressionMismatchError, UndefinedCaseError


class ScalarExpr(Expr, ABC):
    """
    ScalarExpr class and its properties
    """

    def __init__(self):
        super().__init__()
        self.size = (1,)
        self.type = Expression.SCALAR

    def __str__(self):
        raise NotImplementedError

    def __add__(self, other):
        from geomech.operations.addition import Add
        return Add(self, other)

    def __iadd__(self, other):
        from geomech.operations.addition import Add
        return Add(self, other)

    def __sub__(self, other):
        from geomech.operations.addition import Add
        from geomech.operations.multiplication import Mul
        if other.type == Expression.SCALAR:
            return Add(self, Mul(other, -1))
        else:
            raise ExpressionMismatchError('Add', self.type, other.type)

    def __mul__(self, other):
        from geomech.operations.multiplication import Mul, SVMul, SMMul
        if type(other) == float or type(other) == int:
            other = Scalar('(' + str(other) + ')', value=other, attr=['Constant'])
        if other.type == Expression.SCALAR:
            return Mul(self, other)
        elif other.type == Expression.VECTOR:
            return SVMul(other, self)
        elif other.type == Expression.MATRIX:
            return SMMul(other, self)
        else:
            raise UndefinedCaseError


class Scalar(ScalarExpr, ABC):
    """
    Scalar Variable
    """

    def __init__(self, s=None, value=None, attr=None):
        super().__init__()
        self.name = s
        self.value = value
        self.attr = attr

    def __str__(self):
        return self.name

    def delta(self):
        if self.isConstant:
            return Scalar('0', value=0)
        else:
            from geomech.operations.geometry import Delta
            return Delta(self)

    def variation_vector(self):
        return self.delta()

    def diff(self):
        """differentiation"""
        if self.isConstant:
            return Scalar(s='0', value=0, attr=['Constant', 'Zero'])
        else:
            return Scalar(s='dot_' + self.name)

    def integrate(self):
        if self.isConstant:
            raise NotImplementedError
        else:
            s = self.name
            if 'dot_' in s:
                s.replace('dot_', '')
                return Scalar(s=s)
            else:
                return Scalar(s='int_' + s)

    def has(self, elem):
        return self.name == elem.name


Zero = Scalar('0', value=0, attr=['Constant', 'Zero'])
One = Scalar('1', value=1, attr=['Constant', 'Ones'])


def Number(value):
    if type(value) == float or type(value) == int:
        return Scalar('(' + str(value) + ')', value=value, attr=['Constant'])
    elif isinstance(value, str):
        return Scalar(s=value, attr=['Constant'])
    else:
        raise Exception('Input to number should be int/float/string')


def getScalars(x, attr=None):
    if isinstance(x, list):
        variables = x
    elif isinstance(x, str):
        variables = x.split()
    else:
        return None
    s = []
    for v in variables:
        s.append(Scalar(v, attr=attr))
    return tuple(s)
