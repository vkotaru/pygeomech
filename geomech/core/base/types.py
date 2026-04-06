from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ExprType(Enum):
    SCALAR = 1
    VECTOR = 2
    MATRIX = 3


@dataclass
class ExprFlags:
    is_constant: bool = False
    is_zero: bool = False
    is_ones: bool = False
    is_unit_norm: bool = False
    is_identity: bool = False
    is_symmetric: bool = False
    is_skew_symmetric: bool = False
    is_orthogonal: bool = False
    is_numeric: bool = False
    is_manifold: bool = False

    @classmethod
    def from_attr_list(cls, attrs: list[str] | None) -> ExprFlags:
        if not attrs:
            return cls()
        flags = cls()
        for a in attrs:
            match a:
                case 'Constant':
                    flags.is_constant = True
                case 'Zero':
                    flags.is_zero = True
                    flags.is_constant = True
                case 'Ones':
                    flags.is_ones = True
                    flags.is_constant = True
                case 'Identity':
                    flags.is_identity = True
                    flags.is_constant = True
                case 'UnitNorm':
                    flags.is_unit_norm = True
                case 'SymmetricMatrix':
                    flags.is_symmetric = True
                case 'SkewSymmetricMatrix':
                    flags.is_skew_symmetric = True
                case 'OrthogonalMatrix':
                    flags.is_orthogonal = True
                case 'Manifold':
                    flags.is_manifold = True
        return flags


@dataclass
class ManifoldInfo:
    tangent_vector_name: str
    variation_vector_name: str
