from abc import ABC

import numpy as np
from geomech.base.expr import Expression, Expr, Manifold
from geomech.utils.errors import ExpressionMismatchError, UndefinedCaseError


class VectorExpr(Expr, ABC):
    def __init__(self):
        super().__init__()
        self.type = Expression.VECTOR

    def __str__(self):
        raise NotImplementedError

    def __add__(self, other):
        from geomech.operations.addition import VAdd
        return VAdd(self, other)

    def __iadd__(self, other):
        from geomech.operations.addition import VAdd
        return VAdd(self, other)

    def __sub__(self, other):
        from geomech.operations.addition import VAdd
        from geomech.operations.multiplication import SVMul
        if other.type == Expression.VECTOR:
            return VAdd(self, SVMul(other, -1))
        else:
            raise ExpressionMismatchError('Sub', self.type, other.type)

    def __mul__(self, other):
        from geomech.operations.multiplication import SVMul, VVMul, MVMul
        from geomech.base.scalars import Scalar
        if type(other) == float or type(other) == int:
            other = Scalar('(' + str(other) + ')', value=other, attr=['Constant'])
        if other.type == Expression.SCALAR:
            return SVMul(self, other)
        elif other.type == Expression.VECTOR:
            return VVMul(self, other)
        elif other.type == Expression.MATRIX:
            return MVMul(self, other)
        else:
            raise UndefinedCaseError

    def dot(self, other):
        from geomech.operations.geometry import Dot
        return Dot(self, other)

    def cross(self, other):
        from geomech.operations.geometry import Cross
        return Cross(self, other)

    def T(self):
        from geomech.operations.transpose import Transpose
        return Transpose(self)


class Vector(VectorExpr, ABC):
    def __init__(self, s, size=(3,), value=None, attr=None):
        super().__init__()
        self.name = s
        self.size = size
        if value is None:
            self.value = np.empty(size, dtype='object')
        else:
            self.value = value
            self.size = self.value.shape
        self.attr = attr  # Constant, Zero, Ones

    def __str__(self):
        return self.name

    def delta(self):
        if self.isOnes or self.isZero or self.isConstant:
            return Vector('0', attr=['Constant', 'Zero'])
        else:
            from geomech.operations.geometry import Delta
            return Delta(self)

    def diff(self):
        if self.isConstant:
            return Vector(s='0', size=self.size, attr=['Constant', 'Zero'])
        else:
            return Vector(s='dot_' + self.name, size=self.size)

    def get_variation_vector(self):
        return self.delta()

    def integrate(self):
        if self.isConstant:
            raise NotImplementedError
        else:
            s = self.name
            if 'dot_' in s:
                new_s = s.replace("dot_", "")
                return Vector(s=new_s, size=self.size)
            else:
                return Vector(s='int_' + s, size=self.size)


class TSO3(Vector, ABC):
    """TSO(3) Tangent space of SO3 manifold"""

    def __init__(self, s, SO3=None):
        super().__init__(s, value=None, attr=None)
        self.SO3 = SO3
        self.attr.append('TangentVector')

    def delta(self, substitute=False):
        """
        # reference: https://link.springer.com/book/10.1007%2F978-3-319-56953-6
        # Global Formulations of Lagrangian and Hamiltonian Dynamics on Manifolds
        # Taeyoung LeeMelvin LeokN. Harris McClamroch
        # TODO add page number
        """
        from geomech.operations.geometry import Hat, Delta
        if substitute:
            eta = self.SO3.get_variation_vector()
            return Hat(self) * eta + eta.diff()
        else:
            return Delta(self)


class TS2(Vector, ABC):
    """TS2 Tangent space of S2 manifold"""

    def __init__(self, s, S2=None):
        super().__init__(s, value=None, attr=None)
        self.S2 = S2
        self.attr.append('TangentVector')

    def delta(self, substitute=False):
        """
        # reference: https://link.springer.com/book/10.1007%2F978-3-319-56953-6
        # Global Formulations of Lagrangian and Hamiltonian Dynamics on Manifolds
        # Taeyoung LeeMelvin LeokN. Harris McClamroch
        # TODO add page number
        """
        from geomech.operations.multiplication import VVMul
        from geomech.operations.geometry import Hat, Delta
        from geomech.operations.transpose import Transpose
        from geomech.base.matrices import I
        if substitute:
            raise NotImplementedError
        else:
            return Delta(self)


class S2(Vector, Manifold, ABC):
    def __init__(self, s=None, size=(3, 1), value=None, attr=None):
        super().__init__(s, size, value, attr)
        super(Manifold, self).__init__()
        self.tangent_vector = '\\omega_{' + self.name + '}'
        self.variation_vector = '\\xi_{' + self.name + '}'
        if attr is None:
            attr = []
        attr.append('Manifold')
        self.attr = attr

    def delta(self):
        from geomech.operations.geometry import Cross
        return Cross(self.get_variation_vector(), self)

    def get_tangent_vector(self):
        return TS2(self.tangent_vector, S2=self)

    def get_variation_vector(self):
        return Vector(self.variation_vector)

    def diff(self):
        from geomech.operations.geometry import Cross
        return Cross(self.get_tangent_vector(), self)


ZeroVector = Vector(s='0v', attr=['Constant', 'Zero'])


def getVectors(x, attr=None):
    if isinstance(x, list):
        variables = x
    elif isinstance(x, str):
        variables = x.split()
    else:
        return None
    s = []
    for v in variables:
        s.append(Vector(v, attr=attr))
    return tuple(s)
