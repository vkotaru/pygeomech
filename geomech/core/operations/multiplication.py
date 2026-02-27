from __future__ import annotations

from dataclasses import dataclass, field

from geomech.core.base.expressions import (
    Expr, ScalarExpr, VectorExpr, MatrixExpr, Scalar,
)
from geomech.core.operations.mixins import _BinaryMixin
from geomech.core.base.types import ExprType
from geomech.utils.errors import ExpressionMismatchError, SizeMismatchError


# ---------------------------------------------------------------------------
# Binary multiplication helpers
# ---------------------------------------------------------------------------

def _wrap_numeric(val):
    """Wrap int/float to Scalar constant."""
    if isinstance(val, (int, float)):
        return Scalar('(' + str(val) + ')', value=val, attr=['Constant'])
    return val


def _check_mul_sizes(op_name: str, l, r):
    """Validate matrix/vector dimension compatibility for multiplication.

    Rules (when both operands carry a known size):
      MVMul: M(m,n) * v(p,) → n must equal p
      MMMul: M(m,n) * N(p,q) → n must equal p
      VVMul (inner): v(m,)^T * w(p,) → m must equal p
    """
    ls = getattr(l, 'size', None)
    rs = getattr(r, 'size', None)
    if ls is None or rs is None:
        return
    # columns of left must match rows/length of right
    l_cols = ls[-1]          # last dim: cols for matrix, length for vector
    r_rows = rs[0]           # first dim: rows for matrix, length for vector
    if l_cols != r_rows:
        raise SizeMismatchError(
            op_name, 'left cols=' + str(l_cols), 'right rows=' + str(r_rows)
        )


# ---------------------------------------------------------------------------
# Scalar * Scalar
# ---------------------------------------------------------------------------

@dataclass(eq=False)
class Mul(_BinaryMixin, ScalarExpr):
    """Scalar multiplication."""
    nodes: list = field(default_factory=list)

    def __init__(self, l, r):
        l, r = _wrap_numeric(l), _wrap_numeric(r)
        if l.type == ExprType.SCALAR and r.type == ExprType.SCALAR:
            self.nodes = [l, r]
        else:
            raise ExpressionMismatchError('Mul', l.type, r.type)

    def __str__(self):
        return str(self.left) + str(self.right)


# ---------------------------------------------------------------------------
# Scalar * Vector
# ---------------------------------------------------------------------------

@dataclass(eq=False)
class SVMul(_BinaryMixin, VectorExpr):
    """Scalar-Vector multiplication.  Normalized: left=vector, right=scalar."""
    nodes: list = field(default_factory=list)

    def __init__(self, l, r):
        l, r = _wrap_numeric(l), _wrap_numeric(r)
        if l.type == ExprType.VECTOR and r.type == ExprType.SCALAR:
            self.nodes = [l, r]
        elif l.type == ExprType.SCALAR and r.type == ExprType.VECTOR:
            self.nodes = [r, l]
        else:
            raise ExpressionMismatchError('SVMul', l.type, r.type)

    def __str__(self):
        return str(self.left) + str(self.right)


# ---------------------------------------------------------------------------
# Scalar * Matrix
# ---------------------------------------------------------------------------

@dataclass(eq=False)
class SMMul(_BinaryMixin, MatrixExpr):
    """Scalar-Matrix multiplication.  Normalized: left=matrix, right=scalar."""
    nodes: list = field(default_factory=list)

    def __init__(self, l, r):
        l, r = _wrap_numeric(l), _wrap_numeric(r)
        if l.type == ExprType.MATRIX and r.type == ExprType.SCALAR:
            self.nodes = [l, r]
        elif l.type == ExprType.SCALAR and r.type == ExprType.MATRIX:
            self.nodes = [r, l]
        else:
            raise ExpressionMismatchError('SMMul', l.type, r.type)

    def __str__(self):
        return str(self.left) + str(self.right)


# ---------------------------------------------------------------------------
# Matrix * Vector
# ---------------------------------------------------------------------------

@dataclass(eq=False)
class MVMul(_BinaryMixin, VectorExpr):
    """Matrix-Vector multiplication."""
    nodes: list = field(default_factory=list)

    def __init__(self, l, r):
        if l.type == ExprType.MATRIX and r.type == ExprType.VECTOR:
            _check_mul_sizes('MVMul', l, r)
            self.nodes = [l, r]
        elif l.type == ExprType.VECTOR and r.type == ExprType.MATRIX:
            # Transpose(v) * M case
            from geomech.core.operations.geometry import Transpose
            if isinstance(l, Transpose):
                self.nodes = [l, r]
            else:
                raise ExpressionMismatchError('MVMul', l.type, r.type)
        else:
            raise ExpressionMismatchError('MVMul', l.type, r.type)

    def __str__(self):
        return str(self.left) + str(self.right)


# ---------------------------------------------------------------------------
# Matrix * Matrix
# ---------------------------------------------------------------------------

@dataclass(eq=False)
class MMMul(_BinaryMixin, MatrixExpr):
    """Matrix-Matrix multiplication."""
    nodes: list = field(default_factory=list)

    def __init__(self, l, r):
        if l.type == ExprType.MATRIX and r.type == ExprType.MATRIX:
            _check_mul_sizes('MMMul', l, r)
            self.nodes = [l, r]
        else:
            raise ExpressionMismatchError('MMMul', l.type, r.type)

    def __str__(self):
        return str(self.left) + str(self.right)


# ---------------------------------------------------------------------------
# Vector * Vector
# ---------------------------------------------------------------------------

@dataclass(eq=False)
class VVMul(_BinaryMixin, Expr):
    """Vector-Vector multiplication.
    Result type depends on Transpose:
      Transpose(v) * w  → scalar
      v * Transpose(w)  → matrix
    """
    nodes: list = field(default_factory=list)

    def __init__(self, l, r):
        from geomech.core.operations.geometry import Transpose
        if l.type == ExprType.VECTOR and r.type == ExprType.VECTOR:
            if isinstance(l, Transpose) and not isinstance(r, Transpose):
                # v^T * w → scalar: sizes must match
                _check_mul_sizes('VVMul', l, r)
                self.nodes = [l, r]
                self._result_type = ExprType.SCALAR
            elif not isinstance(l, Transpose) and isinstance(r, Transpose):
                # v * w^T → matrix (outer product): any sizes valid
                self.nodes = [l, r]
                self._result_type = ExprType.MATRIX
            else:
                raise ExpressionMismatchError('VVMul', l.type, r.type)
        else:
            raise ExpressionMismatchError('VVMul', l.type, r.type)

    @property
    def type(self):
        return self._result_type

    def __str__(self):
        return str(self.left) + str(self.right)
