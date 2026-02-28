from __future__ import annotations

from geomech.core.base.types import ExprType
from geomech.utils.errors import ExpressionMismatchError, UndefinedCaseError


# ---------------------------------------------------------------------------
# Base — shared by all operation nodes
# ---------------------------------------------------------------------------

class _BaseMixin:
    """Shared base for all operation nodes."""

    def has(self, elem):
        if self == elem:
            return True
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
    def is_constant(self):
        return all(n.is_constant for n in self.nodes)

    @property
    def is_zero(self):
        return all(n.is_zero for n in self.nodes)

    def __str__(self):
        return '(' + '+'.join(str(n) for n in self.nodes) + ')'

    def delta(self):
        from geomech.core.base.expressions import Zero, ZeroVector, ZeroMatrix
        if self.is_constant:
            match self.type:
                case ExprType.SCALAR: return Zero
                case ExprType.VECTOR: return ZeroVector
                case ExprType.MATRIX: return ZeroMatrix
        return type(self)(*[n.delta() for n in self.nodes])

    def t_diff(self):
        from geomech.core.base.expressions import Zero, ZeroVector, ZeroMatrix
        if self.is_constant:
            match self.type:
                case ExprType.SCALAR: return Zero
                case ExprType.VECTOR: return ZeroVector
                case ExprType.MATRIX: return ZeroMatrix
        return type(self)(*[n.t_diff() for n in self.nodes])


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
    def is_constant(self):
        return self.left.is_constant and self.right.is_constant

    @property
    def is_zero(self):
        return self.left.is_zero or self.right.is_zero

    def _apply_rule(self, op_name):
        """Apply linearity rule (delta or t_diff) with product-rule logic.

        op_name: 'delta' or 't_diff'
        """
        from geomech.core.base.expressions import Zero, ZeroVector, ZeroMatrix
        if self.is_constant:
            match self.type:
                case ExprType.SCALAR: return Zero
                case ExprType.VECTOR: return ZeroVector
                case ExprType.MATRIX: return ZeroMatrix
        op = lambda node: getattr(node, op_name)()
        if self.left.is_constant:
            return type(self)(self.left, op(self.right))
        if self.right.is_constant:
            return type(self)(op(self.left), self.right)
        # Product rule: op(l*r) = op(l)*r + l*op(r)
        term1 = type(self)(op(self.left), self.right)
        term2 = type(self)(self.left, op(self.right))
        from geomech.core.operations.addition import Add, VAdd, MAdd
        match self.type:
            case ExprType.SCALAR: return Add(term1, term2)
            case ExprType.VECTOR: return VAdd(term1, term2)
            case ExprType.MATRIX: return MAdd(term1, term2)

    def delta(self):
        return self._apply_rule('delta')

    def t_diff(self):
        return self._apply_rule('t_diff')


# ---------------------------------------------------------------------------
# Unary (fixed output type — Hat, Vee, etc.)
# ---------------------------------------------------------------------------

class _UnaryMixin(_BaseMixin):
    """Shared properties for unary operation nodes with fixed output type."""

    @property
    def expr(self):
        return self.nodes[0]

    @property
    def is_constant(self):
        return self.expr.is_constant

    @property
    def is_zero(self):
        return self.expr.is_zero

    def delta(self):
        from geomech.core.base.expressions import Zero, ZeroVector, ZeroMatrix
        if self.is_constant:
            match self.type:
                case ExprType.SCALAR: return Zero
                case ExprType.VECTOR: return ZeroVector
                case ExprType.MATRIX: return ZeroMatrix
        return type(self)(self.expr.delta())

    def t_diff(self):
        from geomech.core.base.expressions import Zero, ZeroVector, ZeroMatrix
        if self.is_constant:
            match self.type:
                case ExprType.SCALAR: return Zero
                case ExprType.VECTOR: return ZeroVector
                case ExprType.MATRIX: return ZeroMatrix
        return type(self)(self.expr.t_diff())


# ---------------------------------------------------------------------------
# Calculus unary (Variation, TimeDerivative, TimeIntegral, Delta)
# ---------------------------------------------------------------------------

class _CalcUnaryMixin(_UnaryMixin):
    """Type-preserving unary — delegates type to inner expression.

    Pushes t_integrate() inside by default so that e.g.
    Variation(expr).t_integrate() → Variation(expr.t_integrate()).
    Subclasses (TimeDerivative, TimeIntegral) override for cancellation.
    """

    @property
    def type(self):
        return self.expr.type

    def t_integrate(self):
        from geomech.core.base.expressions import Zero, ZeroVector, ZeroMatrix
        if self.is_constant:
            match self.type:
                case ExprType.SCALAR: return Zero
                case ExprType.VECTOR: return ZeroVector
                case ExprType.MATRIX: return ZeroMatrix
        return type(self)(self.expr.t_integrate())
