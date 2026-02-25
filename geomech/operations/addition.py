from abc import ABC

from geomech.base.matrices import MatrixExpr, ZeroMatrix
from geomech.base.scalars import ScalarExpr, Zero, Scalar
from geomech.base.vectors import VectorExpr, ZeroVector
from geomech.base.nodes import NaryNode


class Add(NaryNode, ScalarExpr, ABC):
    """Scalar Addition"""

    def __init__(self, *args):
        super().__init__(*args)

    def __str__(self):
        return super().get_str('+')

    def delta(self):
        """variation"""
        deltas = []
        for n in self.nodes:
            if not n.isConstant:
                deltas.append(n.delta())
        if len(deltas) > 0:
            return Add(deltas)
        else:
            return Zero

    def diff(self):
        """
        time differentiation
        # TODO delta and diff follow same struct; combine them in NaryNode
        """
        diffs = []
        for n in self.nodes:
            if not n.isConstant:
                diffs.append(n.diff())
        if len(diffs) > 0:
            return Add(diffs)
        else:
            return Zero


class VAdd(NaryNode, VectorExpr, ABC):
    """Vector Addition"""

    def __init__(self, *args):
        super().__init__(*args)

    def __str__(self):
        return super().get_str('+')

    def delta(self):
        """variation"""
        deltas = []
        for n in self.nodes:
            if not n.isConstant:
                deltas.append(n.delta())
        if len(deltas) > 0:
            return VAdd(deltas)
        else:
            return ZeroVector

    def diff(self):
        """
        time differentiation
        # TODO delta and diff follow same struct; combine them in NaryNode
        """
        diffs = []
        for n in self.nodes:
            if not n.isConstant:
                diffs.append(n.diff())
        if len(diffs) > 0:
            return VAdd(diffs)
        else:
            return ZeroVector


class MAdd(NaryNode, MatrixExpr, ABC):
    """Matrix Addition"""

    def __init__(self, *args):
        super().__init__(*args)

    def __str__(self):
        return super().get_str('+')

    def delta(self):
        """variation"""
        deltas = []
        for n in self.nodes:
            if not n.isConstant:
                deltas.append(n.delta())
        if len(deltas) > 0:
            return MAdd(deltas)
        else:
            return ZeroMatrix

    def diff(self):
        """
        time differentiation
        # TODO delta and diff follow same struct; combine them in NaryNode
        """
        diffs = []
        for n in self.nodes:
            if not n.isConstant:
                diffs.append(n.diff())
        if len(diffs) > 0:
            return MAdd(diffs)
        else:
            return ZeroMatrix
