"""
geomech is a python library for symbolic computation of Dynamics on Manifolds.
"""

import sys

if sys.version_info < (3, 10):
    raise ImportError("Python version >= 3.10 is required")
del sys

# --- Layer 1: types & leaf nodes ---
from geomech.core.base import (
    S2,
    SO3,
    TS2,
    TSO3,
    Expr,
    ExprFlags,
    ExprType,
    I,
    IdentityMatrix,
    ManifoldInfo,
    Matrix,
    MatrixExpr,
    Number,
    O,
    One,
    Scalar,
    ScalarExpr,
    SkewSymmMatrix,
    Vector,
    VectorExpr,
    Zero,
    ZeroMatrix,
    ZeroVector,
    getMatrices,
    getScalars,
    getVectors,
)

# --- Layer 4: symbolic math ---
from geomech.core.math import (
    collect,
    extract_coeff,
    integrate_by_parts,
)

# --- Layer 2: expression tree nodes ---
from geomech.core.operations import (
    Add,
    Cross,
    Dot,
    Hat,
    MAdd,
    MMMul,
    Mul,
    MVMul,
    SMMul,
    SVMul,
    TimeDerivative,
    TimeIntegral,
    Transpose,
    VAdd,
    Variation,
    Vee,
    VVMul,
)

# --- Layer 3: algorithms ---
from geomech.core.transformations import (
    expand,
    full_simplify,
    has_nested_add,
    is_leaf,
    pull,
    simplify,
    vector_rules,
)

# --- Layer 5: dynamics pipeline ---
from geomech.dynamics import (
    StandardFormEquation,
    SystemVariables,
    compute_eom,
    separate_variations,
    to_standard_form,
)

# --- Utilities ---
from geomech.utils import (
    display_eom,
    display_latex,
    display_standard_form,
    eom_to_latex,
    print_eom,
    print_tree,
    render_eom,
    to_latex,
    tree_str,
)
