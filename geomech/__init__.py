"""
geomech is a python library for symbolic computation of Dynamics on Manifolds.
"""

import sys

if sys.version_info < (3, 10):
    raise ImportError("Python version >= 3.10 is required")
del sys

# --- Layer 1: types & leaf nodes ---
from geomech.core.base import (
    ExprType, ExprFlags, ManifoldInfo,
    Expr, ScalarExpr, Scalar, VectorExpr, Vector, MatrixExpr, Matrix,
    S2, TS2, TSO3, SO3, SkewSymmMatrix,
    Zero, One, ZeroVector, ZeroMatrix, IdentityMatrix, O, I,
    Number, getScalars, getVectors, getMatrices,
)

# --- Layer 2: expression tree nodes ---
from geomech.core.operations import (
    Add, VAdd, MAdd,
    Mul, SVMul, SMMul, MVMul, MMMul, VVMul,
    Dot, Cross, Hat, Vee, Transpose,
    Variation, TimeDerivative, TimeIntegral,
)

# --- Layer 3: algorithms ---
from geomech.core.transformations import (
    expand, pull, vector_rules, simplify, full_simplify,
    is_leaf, has_nested_add,
)

# --- Layer 4: symbolic math ---
from geomech.core.math import (
    extract_coeff, collect, integrate_by_parts,
)

# --- Layer 5: dynamics pipeline ---
from geomech.dynamics import SystemVariables, compute_eom, separate_variations

# --- Utilities ---
from geomech.utils import print_eom, print_tree, tree_str, render_eom, eom_to_latex
