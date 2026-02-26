from __future__ import annotations

from geomech.core.types import ExprType
from geomech.utils.errors import ExpressionMismatchError, UndefinedCaseError


# ---------------------------------------------------------------------------
# Base — shared by all operation nodes
# ---------------------------------------------------------------------------

class _BaseMixin:
    """Shared base for all operation nodes."""

    def has(self, elem):
        return any(n.has(elem) for n in self.nodes)

    @property
    def arity(self):
        return len(self.nodes)

    def __len__(self):
        return len(self.nodes)

    # ------ arithmetic dispatch (centralised for all operation nodes) ------

    def __add__(self, other):
        from geomech.core.operations.addition import Add, VAdd, MAdd
        match self.type:
            case ExprType.SCALAR:
                return Add(self, other)
            case ExprType.VECTOR:
                return VAdd(self, other)
            case ExprType.MATRIX:
                return MAdd(self, other)
            case _:
                raise UndefinedCaseError

    def __iadd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        from geomech.core.operations.addition import Add, VAdd, MAdd
        from geomech.core.operations.multiplication import Mul, SVMul, SMMul
        match self.type:
            case ExprType.SCALAR:
                if other.type != ExprType.SCALAR:
                    raise ExpressionMismatchError('Sub', self.type, other.type)
                return Add(self, Mul(other, -1))
            case ExprType.VECTOR:
                if other.type != ExprType.VECTOR:
                    raise ExpressionMismatchError('Sub', self.type, other.type)
                return VAdd(self, SVMul(other, -1))
            case ExprType.MATRIX:
                if other.type != ExprType.MATRIX:
                    raise ExpressionMismatchError('Sub', self.type, other.type)
                return MAdd(self, SMMul(other, -1))
            case _:
                raise UndefinedCaseError

    def __mul__(self, other):
        from geomech.core.operations.multiplication import (
            Mul, SVMul, SMMul, VVMul, MVMul, MMMul, _wrap_numeric,
        )
        from geomech.core.operations.geometry import Transpose
        other = _wrap_numeric(other)
        match (self.type, other.type):
            case (ExprType.SCALAR, ExprType.SCALAR):
                return Mul(self, other)
            case (ExprType.SCALAR, ExprType.VECTOR):
                return SVMul(other, self)
            case (ExprType.SCALAR, ExprType.MATRIX):
                return SMMul(other, self)
            case (ExprType.VECTOR, ExprType.SCALAR):
                return SVMul(self, other)
            case (ExprType.VECTOR, ExprType.VECTOR):
                return VVMul(self, other)
            case (ExprType.VECTOR, ExprType.MATRIX):
                return MVMul(self, other)
            case (ExprType.MATRIX, ExprType.SCALAR):
                return SMMul(self, other)
            case (ExprType.MATRIX, ExprType.VECTOR):
                if isinstance(other, Transpose):
                    raise ExpressionMismatchError(
                        type(self).__name__ + '.__mul__', self.type, other.type
                    )
                return MVMul(self, other)
            case (ExprType.MATRIX, ExprType.MATRIX):
                return MMMul(self, other)
            case _:
                raise UndefinedCaseError


# ---------------------------------------------------------------------------
# N-ary (addition)
# ---------------------------------------------------------------------------

class _NaryMixin(_BaseMixin):
    """Shared properties for n-ary addition operations."""

    @property
    def isConstant(self):
        return all(n.isConstant for n in self.nodes)

    @property
    def isZero(self):
        return all(n.isZero for n in self.nodes)

    def __str__(self):
        return '(' + '+'.join(str(n) for n in self.nodes) + ')'


# ---------------------------------------------------------------------------
# Binary (multiplication)
# ---------------------------------------------------------------------------

class _BinaryMixin(_BaseMixin):
    """Shared properties for binary operations using unified nodes."""

    @property
    def left(self):
        return self.nodes[0]

    @property
    def right(self):
        return self.nodes[1]

    @property
    def isConstant(self):
        return self.left.isConstant and self.right.isConstant

    @property
    def isZero(self):
        return self.left.isZero or self.right.isZero


# ---------------------------------------------------------------------------
# Calculus unary (Variation, TimeDerivative, TimeIntegral)
# ---------------------------------------------------------------------------

class _CalcUnaryMixin(_BaseMixin):
    """Shared properties for type-preserving calculus unary nodes."""

    @property
    def expr(self):
        return self.nodes[0]

    @property
    def type(self):
        return self.expr.type

    @property
    def isConstant(self):
        return self.expr.isConstant

    @property
    def isZero(self):
        return self.expr.isZero
