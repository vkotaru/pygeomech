from __future__ import annotations

from dataclasses import dataclass, field

from geomech.core.base.expressions import (
    MatrixExpr,
    ScalarExpr,
    VectorExpr,
)
from geomech.core.base.types import ExprType
from geomech.core.operations.mixins import _NaryMixin
from geomech.utils.errors import ExpressionMismatchError, SizeMismatchError

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _check_sizes(nodes, op_name: str):
    """Validate that all operands with a known size are compatible."""
    ref_size = None
    for n in nodes:
        s = getattr(n, "size", None)
        if s is not None:
            if ref_size is None:
                ref_size = s
            elif s != ref_size:
                raise SizeMismatchError(op_name, ref_size, s)


def _flatten_nodes(args, expected_type: type, expr_type: ExprType) -> list:
    """Flatten nested same-type adds and validate types."""
    nodes = []
    for arg in args:
        if isinstance(arg, (list, tuple)):
            for a in arg:
                if a.type != expr_type:
                    raise ExpressionMismatchError(expected_type.__name__, expr_type, a.type)
                nodes.append(a)
        elif isinstance(arg, expected_type):
            nodes.extend(arg.nodes)
        else:
            if arg.type != expr_type:
                raise ExpressionMismatchError(expected_type.__name__, expr_type, arg.type)
            nodes.append(arg)
    # Size validation for vector/matrix additions
    if expr_type in (ExprType.VECTOR, ExprType.MATRIX):
        _check_sizes(nodes, expected_type.__name__)
    return nodes


# ---------------------------------------------------------------------------
# N-ary addition
# ---------------------------------------------------------------------------


@dataclass(eq=False, repr=False)
class Add(_NaryMixin, ScalarExpr):
    """Scalar addition (n-ary)."""

    nodes: list = field(default_factory=list)

    def __init__(self, *args):
        self.nodes = _flatten_nodes(args, Add, ExprType.SCALAR)


@dataclass(eq=False, repr=False)
class VAdd(_NaryMixin, VectorExpr):
    """Vector addition (n-ary)."""

    nodes: list = field(default_factory=list)

    def __init__(self, *args):
        self.nodes = _flatten_nodes(args, VAdd, ExprType.VECTOR)


@dataclass(eq=False, repr=False)
class MAdd(_NaryMixin, MatrixExpr):
    """Matrix addition (n-ary)."""

    nodes: list = field(default_factory=list)

    def __init__(self, *args):
        self.nodes = _flatten_nodes(args, MAdd, ExprType.MATRIX)
