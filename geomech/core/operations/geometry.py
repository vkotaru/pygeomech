from __future__ import annotations

from dataclasses import dataclass, field

from geomech.core.base.expressions import (
    Expr, ScalarExpr, VectorExpr, MatrixExpr,
)
from geomech.core.operations.mixins import _BinaryMixin, _UnaryMixin, _BaseMixin
from geomech.core.base.types import ExprType
from geomech.utils.errors import ExpressionMismatchError, SizeMismatchError



# ---------------------------------------------------------------------------
# Dot  (scalar result from two vectors)
# ---------------------------------------------------------------------------

@dataclass(eq=False)
class Dot(_BinaryMixin, ScalarExpr):
    """Dot product of two vectors → scalar."""
    nodes: list = field(default_factory=list)

    def __init__(self, l, r):
        if l.type == ExprType.VECTOR and r.type == ExprType.VECTOR:
            ls = getattr(l, 'size', None)
            rs = getattr(r, 'size', None)
            if ls is not None and rs is not None and ls != rs:
                raise SizeMismatchError('Dot', ls, rs)
            self.nodes = [l, r]
        else:
            raise ExpressionMismatchError('Dot', l.type, r.type)

    def __str__(self):
        return 'Dot(' + str(self.left) + ',' + str(self.right) + ')'


# ---------------------------------------------------------------------------
# Cross  (vector result from two vectors)
# ---------------------------------------------------------------------------

@dataclass(eq=False)
class Cross(_BinaryMixin, VectorExpr):
    """Cross product of two 3-vectors → vector."""
    nodes: list = field(default_factory=list)

    def __init__(self, l, r):
        if l.type == ExprType.VECTOR and r.type == ExprType.VECTOR:
            ls = getattr(l, 'size', None)
            rs = getattr(r, 'size', None)
            if ls is not None and rs is not None and ls != rs:
                raise SizeMismatchError('Cross', ls, rs)
            self.nodes = [l, r]
        else:
            raise ExpressionMismatchError('Cross', l.type, r.type)

    def __str__(self):
        return 'Cross(' + str(self.left) + ',' + str(self.right) + ')'


# ---------------------------------------------------------------------------
# Hat  (vector → skew-symmetric matrix)
# ---------------------------------------------------------------------------

@dataclass(eq=False)
class Hat(_UnaryMixin, MatrixExpr):
    """Hat map: R^3 → so(3)."""
    nodes: list = field(default_factory=list)

    def __init__(self, expr):
        if expr.type == ExprType.VECTOR:
            self.nodes = [expr]
        else:
            raise ExpressionMismatchError('Hat', ExprType.VECTOR, expr.type)

    def __str__(self):
        return 'Hat(' + str(self.expr) + ')'


# ---------------------------------------------------------------------------
# Vee  (skew-symmetric matrix → vector)
# ---------------------------------------------------------------------------

@dataclass(eq=False)
class Vee(_UnaryMixin, VectorExpr):
    """Vee map: so(3) → R^3."""
    nodes: list = field(default_factory=list)

    def __init__(self, expr):
        if expr.type == ExprType.MATRIX:
            self.nodes = [expr]
        else:
            raise ExpressionMismatchError('Vee', ExprType.MATRIX, expr.type)

    def __str__(self):
        return 'Vee(' + str(self.expr) + ')'


# ---------------------------------------------------------------------------
# Transpose  (type-preserving unary)
# ---------------------------------------------------------------------------

@dataclass(eq=False)
class Transpose(_BaseMixin, Expr):
    """Transpose operator.  Preserves the type of its inner expression."""
    nodes: list = field(default_factory=list)

    def __init__(self, expr=None):
        if expr is not None:
            self.nodes = [expr]
        else:
            self.nodes = []

    @property
    def expr(self):
        return self.nodes[0] if self.nodes else None

    @property
    def type(self):
        return self.expr.type if self.expr else None

    @property
    def is_constant(self):
        return self.expr.is_constant if self.expr else False

    @property
    def is_zero(self):
        return self.expr.is_zero if self.expr else False

    @property
    def size(self):
        return getattr(self.expr, 'size', None)

    def __str__(self):
        return '(' + str(self.expr) + ")\'"

    def delta(self):
        return Transpose(self.expr.delta())

    def t_diff(self):
        return Transpose(self.expr.t_diff())

    def __mul__(self, other):
        from geomech.core.operations.multiplication import SVMul, VVMul, MVMul, _wrap_numeric
        other = _wrap_numeric(other)
        match other.type:
            case ExprType.SCALAR:
                return Transpose(SVMul(self, other))
            case ExprType.VECTOR:
                return VVMul(self, other)
            case ExprType.MATRIX:
                return MVMul(self, other)
            case _:
                raise ExpressionMismatchError('Transpose.__mul__', self.type, other.type)
