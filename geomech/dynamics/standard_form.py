"""Standard form extraction: M(q)*a + f(q,dq) + G(q,dq)*u = 0.

to_standard_form(eom_dict, variables, inputs)
    Decompose EOM into mass matrix, nonlinear terms, and input influence.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from geomech.core.base.expressions import Expr, ZeroVector, ZeroMatrix, S2, SO3
from geomech.core.base.types import ExprType
from geomech.core.math.extract import extract_from_vector
from geomech.core.operations.addition import VAdd
from geomech.core.operations.calculus import TimeDerivative
from geomech.core.transformations.expand import expand
from geomech.core.transformations.simplify import full_simplify
from geomech.dynamics.variables import SystemVariables


@dataclass
class StandardFormEquation:
    """Standard form for one EOM equation.

    ``Σ M[a_j] * a_j  +  f  +  Σ G[u_k] * u_k  =  0``

    Attributes
    ----------
    M : dict[str, Expr]
        Mass/inertia coefficient for each acceleration variable.
    f : Expr
        Nonlinear terms (Coriolis, gravity, etc.) — a vector expression.
    G : dict[str, Expr]
        Input influence coefficient for each input variable.
    """

    M: dict = field(default_factory=dict)
    f: Expr = None
    G: dict = field(default_factory=dict)


def to_standard_form(eom_dict, variables: SystemVariables, inputs=None):
    """Decompose EOM into standard manipulator form.

    Parameters
    ----------
    eom_dict : dict
        Output of ``compute_eom()``:
        ``{str(variation_vec): (variation_vec, eom_expr)}``.
    variables : SystemVariables
        Configuration variables (used to derive acceleration variables).
    inputs : list[Expr], optional
        Input variables (forces, torques). Default: empty.

    Returns
    -------
    dict[str, StandardFormEquation]
        Keyed by the same variation vector names as *eom_dict*.
    """
    if inputs is None:
        inputs = []

    # Derive acceleration variables from config variables
    accel_vars = _get_accel_vars(variables)

    result = {}
    for key, (var_vec, eom_expr) in eom_dict.items():
        sf = _extract_single(eom_expr, accel_vars, inputs)
        result[key] = sf

    return result


def _get_accel_vars(variables: SystemVariables):
    """Derive acceleration variables from config variables.

    - Vector x  →  d²x/dt²  =  TimeDerivative(TimeDerivative(x))
    - S2 q      →  dω/dt    =  TimeDerivative(ω_q)
    - SO3 R     →  dΩ/dt    =  TimeDerivative(Ω_R)
    """
    accels = []
    for v in variables.vectors:
        if isinstance(v, S2):
            accels.append(v.get_tangent_vector().t_diff())
        else:
            accels.append(v.t_diff().t_diff())
    for m in variables.matrices:
        if isinstance(m, SO3):
            accels.append(m.get_tangent_vector().t_diff())
        else:
            accels.append(m.t_diff())
    return accels


def _flatten_vadd(expr):
    """Return a flat list of addends from a vector expression."""
    if isinstance(expr, VAdd):
        return list(expr.nodes)
    return [expr]


def _extract_single(eom_expr, accel_vars, inputs):
    """Extract M, f, G from a single EOM vector expression."""
    # Expand to flatten products over sums
    eom_expr = expand(eom_expr)
    terms = _flatten_vadd(eom_expr)

    # Partition terms by what they contain
    accel_terms = []
    input_terms = []
    residual_terms = []

    for term in terms:
        has_accel = any(term.has(a) for a in accel_vars)
        has_input = any(term.has(u) for u in inputs)

        if has_accel:
            accel_terms.append(term)
        elif has_input:
            input_terms.append(term)
        else:
            residual_terms.append(term)

    # Extract M: coefficient of each acceleration variable
    M = {}
    if accel_terms:
        accel_expr = VAdd(*accel_terms) if len(accel_terms) > 1 else accel_terms[0]
        for a in accel_vars:
            if accel_expr.has(a):
                coeff = extract_from_vector(accel_expr, a)
                coeff = full_simplify(coeff)
                M[str(a)] = coeff

    # Extract G: coefficient of each input variable
    G = {}
    if input_terms:
        input_expr = VAdd(*input_terms) if len(input_terms) > 1 else input_terms[0]
        for u in inputs:
            if input_expr.has(u):
                coeff = extract_from_vector(input_expr, u)
                coeff = full_simplify(coeff)
                G[str(u)] = coeff

    # Residual f: everything that's not acceleration or input
    if not residual_terms:
        f = ZeroVector
    elif len(residual_terms) == 1:
        f = full_simplify(residual_terms[0])
    else:
        f = full_simplify(VAdd(*residual_terms))

    return StandardFormEquation(M=M, f=f, G=G)
