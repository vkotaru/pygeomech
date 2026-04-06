from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from geomech.core.base.types import ExprFlags, ExprType, ManifoldInfo
from geomech.utils.errors import ExpressionMismatchError, UndefinedCaseError

# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------


@dataclass(eq=False, repr=False)
class Expr:
    """Base expression node."""

    def __eq__(self, other):
        return str(self) == str(other)

    def __hash__(self):
        return hash(str(self))

    def __str__(self):
        raise NotImplementedError

    def __repr__(self):
        from geomech.utils.printing import repr_str

        return repr_str(self)

    def delta(self):
        from geomech.core.operations.calculus import Variation

        return Variation(self)

    def t_diff(self):
        from geomech.core.operations.calculus import TimeDerivative

        return TimeDerivative(self)

    def t_integrate(self):
        from geomech.core.operations.calculus import TimeIntegral

        return TimeIntegral(self)

    def get_variation_vector(self):
        return self.delta()

    def get_tangent_vector(self):
        raise NotImplementedError

    # --- flag properties (read from flags if present, else False) ---

    def _flag(self, name):
        flags = getattr(self, "flags", None)
        return getattr(flags, name, False) if flags else False

    @property
    def is_constant(self):
        return self._flag("is_constant")

    @property
    def is_zero(self):
        return self._flag("is_zero")

    @property
    def is_ones(self):
        return self._flag("is_ones")

    @property
    def is_unit_norm(self):
        return self._flag("is_unit_norm")

    @property
    def is_numeric(self):
        return self._flag("is_numeric")

    @property
    def is_symmetric(self):
        return self._flag("is_symmetric")

    @property
    def is_skew_symmetric(self):
        return self._flag("is_skew_symmetric")

    @property
    def is_orthogonal(self):
        return self._flag("is_orthogonal")

    @property
    def is_manifold(self):
        return self._flag("is_manifold")

    def has(self, elem):
        return str(elem) == str(self)


# ---------------------------------------------------------------------------
# Scalar
# ---------------------------------------------------------------------------


@dataclass(eq=False, repr=False)
class ScalarExpr(Expr):
    """Base for scalar-typed expressions."""

    @property
    def type(self):
        return ExprType.SCALAR

    @property
    def size(self):
        return (1,)

    def __add__(self, other):
        from geomech.core.operations import Add

        return Add(self, other)

    def __iadd__(self, other):
        from geomech.core.operations import Add

        return Add(self, other)

    def __sub__(self, other):
        from geomech.core.operations import Add, Mul

        if other.type == ExprType.SCALAR:
            return Add(self, Mul(other, -1))
        else:
            raise ExpressionMismatchError("Add", self.type, other.type)

    def __mul__(self, other):
        from geomech.core.operations import Mul, SMMul, SVMul

        if isinstance(other, (int, float)):
            other = Scalar("(" + str(other) + ")", value=other, attr=["Constant"])
        if other.type == ExprType.SCALAR:
            return Mul(self, other)
        elif other.type == ExprType.VECTOR:
            return SVMul(other, self)
        elif other.type == ExprType.MATRIX:
            return SMMul(other, self)
        else:
            raise UndefinedCaseError


@dataclass(eq=False, repr=False)
class Scalar(ScalarExpr):
    name: str = None
    value: Any = None
    attr: list[str] | None = None
    flags: ExprFlags = field(init=False, repr=False, default=None)

    def __post_init__(self):
        self.flags = ExprFlags.from_attr_list(self.attr)
        if self.value is not None:
            self.flags.is_numeric = True

    def __str__(self):
        return self.name

    def delta(self):
        if self.is_constant or self.is_numeric:
            return Zero
        else:
            from geomech.core.operations import Variation

            return Variation(self)

    def t_diff(self):
        if self.is_constant or self.is_numeric:
            return Zero
        from geomech.core.operations.calculus import TimeDerivative

        return TimeDerivative(self)

    def t_integrate(self):
        if self.is_constant:
            raise NotImplementedError
        from geomech.core.operations.calculus import TimeIntegral

        return TimeIntegral(self)

    # support Scalar(s='a') via custom __init__
    def __init__(self, s=None, *, name=None, value=None, attr=None, size=None):
        # 's' is the legacy positional arg; 'name' is the dataclass field
        self.name = s if s is not None else name
        self.value = value
        self.attr = attr
        self.__post_init__()


# ---------------------------------------------------------------------------
# Vector
# ---------------------------------------------------------------------------


@dataclass(eq=False, repr=False)
class VectorExpr(Expr):
    """Base for vector-typed expressions."""

    @property
    def type(self):
        return ExprType.VECTOR

    def __add__(self, other):
        from geomech.core.operations import VAdd

        return VAdd(self, other)

    def __iadd__(self, other):
        from geomech.core.operations import VAdd

        return VAdd(self, other)

    def __sub__(self, other):
        from geomech.core.operations import SVMul, VAdd

        if other.type == ExprType.VECTOR:
            return VAdd(self, SVMul(other, -1))
        else:
            raise ExpressionMismatchError("Sub", self.type, other.type)

    def __mul__(self, other):
        from geomech.core.operations import MVMul, SVMul, VVMul

        if isinstance(other, (int, float)):
            other = Scalar("(" + str(other) + ")", value=other, attr=["Constant"])
        if other.type == ExprType.SCALAR:
            return SVMul(self, other)
        elif other.type == ExprType.VECTOR:
            return VVMul(self, other)
        elif other.type == ExprType.MATRIX:
            return MVMul(self, other)
        else:
            raise UndefinedCaseError

    def dot(self, other):
        from geomech.core.operations import Dot

        return Dot(self, other)

    def cross(self, other):
        from geomech.core.operations import Cross

        return Cross(self, other)

    def T(self):
        from geomech.core.operations import Transpose

        return Transpose(self)


@dataclass(eq=False, repr=False)
class Vector(VectorExpr):
    name: str = None
    size: tuple = (3,)
    value: Any = None
    attr: list[str] | None = None
    flags: ExprFlags = field(init=False, repr=False, default=None)

    def __post_init__(self):
        self.flags = ExprFlags.from_attr_list(self.attr)
        if self.value is None:
            self.value = np.empty(self.size, dtype="object")
        else:
            self.size = self.value.shape

    def __str__(self):
        return self.name

    def delta(self):
        if self.is_ones or self.is_zero or self.is_constant:
            return Vector("0", attr=["Constant", "Zero"])
        else:
            from geomech.core.operations import Variation

            return Variation(self)

    def t_diff(self):
        if self.is_constant:
            return Vector(s="0", size=self.size, attr=["Constant", "Zero"])
        from geomech.core.operations.calculus import TimeDerivative

        return TimeDerivative(self)

    def get_variation_vector(self):
        return self.delta()

    def t_integrate(self):
        if self.is_constant:
            raise NotImplementedError
        from geomech.core.operations.calculus import TimeIntegral

        return TimeIntegral(self)

    # support Vector(s='x') via custom __init__
    def __init__(self, s=None, *, name=None, size=(3,), value=None, attr=None):
        self.name = s if s is not None else name
        self.size = size
        self.value = value
        self.attr = attr
        self.__post_init__()


class TSO3(Vector):
    """Tangent space of SO3 manifold."""

    def __init__(self, s, *, SO3=None):
        super().__init__(s)
        self.SO3 = SO3
        if self.attr is None:
            self.attr = []
        self.attr.append("TangentVector")

    def delta(self, substitute=False):
        from geomech.core.operations import Cross, Variation

        if substitute:
            eta = self.SO3.get_variation_vector()
            # δ(Ω) = η̇ + Ω × η
            return eta.t_diff() + Cross(self, eta)
        else:
            return Variation(self)


class TS2(Vector):
    """Tangent space of S2 manifold."""

    def __init__(self, s, *, S2=None):
        super().__init__(s)
        self.S2 = S2
        if self.attr is None:
            self.attr = []
        self.attr.append("TangentVector")

    def delta(self, substitute=False):
        from geomech.core.operations import Cross, Variation

        if substitute:
            xi = self.S2.get_variation_vector()
            # δ(ω) = ξ̇ - ω × ξ
            return xi.t_diff() - Cross(self, xi)
        else:
            return Variation(self)


class S2(Vector):
    """S2 manifold (unit sphere in R3, ||q|| = 1)."""

    def __init__(self, s=None, *, size=(3,), value=None, attr=None):
        if attr is None:
            attr = []
        attr.append("Manifold")
        attr.append("UnitNorm")
        super().__init__(s, size=size, value=value, attr=attr)
        self.manifold_info = ManifoldInfo(
            tangent_vector_name="\\omega_{" + self.name + "}",
            variation_vector_name="\\xi_{" + self.name + "}",
        )

    def delta(self):
        from geomech.core.operations import Cross

        return Cross(self.get_variation_vector(), self)

    def get_tangent_vector(self):
        return TS2(self.manifold_info.tangent_vector_name, S2=self)

    def get_variation_vector(self):
        return TS2(self.manifold_info.variation_vector_name, S2=self)

    def t_diff(self):
        from geomech.core.operations import Cross

        return Cross(self.get_tangent_vector(), self)


# ---------------------------------------------------------------------------
# Matrix
# ---------------------------------------------------------------------------


@dataclass(eq=False, repr=False)
class MatrixExpr(Expr):
    """Base for matrix-typed expressions."""

    @property
    def type(self):
        return ExprType.MATRIX

    def __add__(self, other):
        from geomech.core.operations import MAdd

        return MAdd(self, other)

    def __iadd__(self, other):
        from geomech.core.operations import MAdd

        return MAdd(self, other)

    def __sub__(self, other):
        from geomech.core.operations import MAdd, SMMul

        if other.type == ExprType.MATRIX:
            return MAdd(self, SMMul(other, -1))
        else:
            raise ExpressionMismatchError("Sub", self.type, other.type)

    def __mul__(self, other):
        from geomech.core.operations import MMMul, MVMul, SMMul, Transpose

        if isinstance(other, (int, float)):
            other = Scalar("(" + str(other) + ")", value=other, attr=["Constant"])
        if other.type == ExprType.SCALAR:
            return SMMul(self, other)
        elif other.type == ExprType.VECTOR:
            if isinstance(other, Transpose):
                raise ExpressionMismatchError
            return MVMul(self, other)
        elif other.type == ExprType.MATRIX:
            return MMMul(self, other)
        else:
            raise UndefinedCaseError


@dataclass(eq=False, repr=False)
class Matrix(MatrixExpr):
    name: str = None
    size: tuple = (3, 3)
    value: Any = None
    attr: list[str] | None = None
    flags: ExprFlags = field(init=False, repr=False, default=None)

    def __post_init__(self):
        self.flags = ExprFlags.from_attr_list(self.attr)
        if self.value is None:
            self.value = np.empty(self.size, dtype="object")

    def __str__(self):
        return self.name

    def delta(self):
        if self.is_ones or self.is_zero or self.is_constant:
            return Matrix("O", attr=["Constant", "Zero"])
        else:
            from geomech.core.operations import Variation

            return Variation(self)

    def t_diff(self):
        if self.is_constant:
            return Matrix(s="0", size=self.size, attr=["Constant", "Zero"])
        from geomech.core.operations.calculus import TimeDerivative

        return TimeDerivative(self)

    def t_integrate(self):
        if self.is_constant:
            raise NotImplementedError
        from geomech.core.operations.calculus import TimeIntegral

        return TimeIntegral(self)

    # support Matrix(s='M') via custom __init__
    def __init__(self, s=None, *, name=None, size=(3, 3), value=None, attr=None):
        self.name = s if s is not None else name
        self.size = size
        self.value = value
        self.attr = attr if attr is not None else []
        if "SymmetricMatrix" in self.attr:
            pass  # handled by ExprFlags
        self.__post_init__()


class SkewSymmMatrix(Matrix):
    def __init__(self, s=None, *, size=(3, 3), value=None, attr=None):
        super().__init__(s, size=size, value=value, attr=attr)
        self.attr.append("SkewSymmetricMatrix")


class SO3(Matrix):
    """SO(3) rotation matrix manifold (R^T R = I, det(R) = 1)."""

    def __init__(self, s=None, *, size=(3, 3), value=None, attr=None):
        if attr is None:
            attr = []
        attr.append("Manifold")
        attr.append("OrthogonalMatrix")
        super().__init__(s, size=size, value=value, attr=attr)
        self.manifold_info = ManifoldInfo(
            tangent_vector_name="\\Omega_{" + self.name + "}",
            variation_vector_name="\\eta_{" + self.name + "}",
        )

    def delta(self):
        from geomech.core.operations import Hat, MMMul

        return MMMul(self, Hat(self.get_variation_vector()))

    def get_tangent_vector(self):
        return TSO3(self.manifold_info.tangent_vector_name, SO3=self)

    def get_variation_vector(self):
        return TSO3(self.manifold_info.variation_vector_name, SO3=self)

    def t_diff(self):
        from geomech.core.operations import Hat, MMMul

        return MMMul(self, Hat(self.get_tangent_vector()))


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

Zero = Scalar("0", value=0, attr=["Constant", "Zero"])
One = Scalar("1", value=1, attr=["Constant", "Ones"])
ZeroVector = Vector(s="0v", attr=["Constant", "Zero"])
ZeroMatrix = Matrix("0", attr=["Constant", "Zero"])
IdentityMatrix = Matrix("I", attr=["Constant", "Identity"])
O = ZeroMatrix
I = IdentityMatrix


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def Number(value):
    if isinstance(value, (int, float)):
        return Scalar("(" + str(value) + ")", value=value, attr=["Constant"])
    elif isinstance(value, str):
        return Scalar(s=value, attr=["Constant"])
    else:
        raise Exception("Input to Number should be int/float/string")


def getScalars(x, attr=None):
    if isinstance(x, list):
        variables = x
    elif isinstance(x, str):
        variables = x.split()
    else:
        return None
    return tuple(Scalar(v, attr=attr) for v in variables)


def getVectors(x, attr=None):
    if isinstance(x, list):
        variables = x
    elif isinstance(x, str):
        variables = x.split()
    else:
        return None
    return tuple(Vector(v, attr=attr) for v in variables)


def getMatrices(x):
    if isinstance(x, list):
        vars_ = x
    elif isinstance(x, str):
        vars_ = x.split()
    else:
        return None
    return tuple(Matrix(v) for v in vars_)
