from __future__ import annotations

from dataclasses import dataclass, field

from geomech.core.base.expressions import Expr
from geomech.core.operations.mixins import _CalcUnaryMixin


# ---------------------------------------------------------------------------
# Variation  (δ operator as a tree node)
# ---------------------------------------------------------------------------

@dataclass(eq=False)
class Variation(_CalcUnaryMixin, Expr):
    """Variation operator δ{expr}.  Preserves the type of its inner expression."""
    nodes: list = field(default_factory=list)

    def __init__(self, expr):
        self.nodes = [expr]

    def __str__(self):
        return '\\delta{' + str(self.expr) + '}'

    def t_diff(self):
        """d/dt(δx) — put TimeDerivative on the outside.

        δ and d/dt commute, but the expression tree produces
        TimeDerivative(Variation(x)) when taking the variation of d/dt(x).
        This override ensures the IBP target matches that structure.
        """
        return TimeDerivative(self)


# ---------------------------------------------------------------------------
# TimeDerivative  (d/dt operator as a tree node)
# ---------------------------------------------------------------------------

@dataclass(eq=False)
class TimeDerivative(_CalcUnaryMixin, Expr):
    """Time derivative d/dt{expr}.  Preserves the type of its inner expression."""
    nodes: list = field(default_factory=list)

    def __init__(self, expr):
        self.nodes = [expr]

    def __str__(self):
        return '\\frac{d}{dt}(' + str(self.expr) + ')'

    def t_integrate(self):
        """∫(d/dt(x)) dt = x — cancellation."""
        return self.expr


# ---------------------------------------------------------------------------
# TimeIntegral  (∫ dt operator as a tree node)
# ---------------------------------------------------------------------------

@dataclass(eq=False)
class TimeIntegral(_CalcUnaryMixin, Expr):
    """Time integral ∫{expr}dt.  Preserves the type of its inner expression."""
    nodes: list = field(default_factory=list)

    def __init__(self, expr):
        self.nodes = [expr]

    def __str__(self):
        return '\\int{' + str(self.expr) + '}dt'

    def t_diff(self):
        """d/dt(∫x dt) = x — cancellation."""
        return self.expr
