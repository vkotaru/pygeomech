"""Equations of motion via the principle of least action.

compute_eom(lagrangian, inf_work, variables)
    variation → simplify → IBP → expand → extract coefficients

separate_variations(expr, variation_vectors)
    extract coefficient of each variation vector from the action integral
"""

from __future__ import annotations

from geomech.core.math.extract import extract_linear_coeff
from geomech.core.math.ibp import integrate_by_parts
from geomech.core.transformations.expand import expand
from geomech.core.transformations.manifold_rules import apply_manifold_rules
from geomech.core.transformations.simplify import full_simplify
from geomech.dynamics.variables import SystemVariables


def compute_eom(lagrangian, inf_work, variables: SystemVariables):
    """Compute equations of motion using the principle of least action.

    1. Take the variation of the Lagrangian: δL
    2. Form the infinitesimal action integral: δS = δL + δW
    3. Apply manifold kinematic substitutions (e.g. δ(ω) for S2)
    4. Simplify
    5. Gather variation vectors and their time derivatives
    6. Integration by parts to move time derivatives off variation vectors
    7. Expand
    8. Extract coefficients of each independent variation vector
    """
    # variation of the Lagrangian
    dL = lagrangian.delta()

    # infinitesimal action integral
    dS = dL + inf_work

    # apply manifold kinematic substitutions
    dS = apply_manifold_rules(dS, variables)

    dS = full_simplify(dS)

    # gather variation vectors and their time derivatives
    variation_vectors = []
    variation_vector_dots = []

    for s in variables.scalars:
        ds = s.delta()  # Variation(scalar) — scalar type
        variation_vectors.append(ds)
        variation_vector_dots.append(ds.t_diff())

    for vec in variables.vectors:
        x = vec.get_variation_vector()
        variation_vectors.append(x)
        variation_vector_dots.append(x.t_diff())

    for mat in variables.matrices:
        x = mat.get_variation_vector()
        variation_vectors.append(x)
        variation_vector_dots.append(x.t_diff())

    # integration by parts
    dS = integrate_by_parts(dS, variation_vector_dots)

    # expand
    dS = expand(dS)

    # extract equations of motion
    return separate_variations(dS, variation_vectors)


def separate_variations(inf_action_integral, variation_vectors):
    """Extract the equation of motion for each variation vector.

    Returns a dict mapping str(variation_vector) → (variation_vector, equation).
    """
    eom = {}
    for vec in variation_vectors:
        dyn_eqn = extract_linear_coeff(inf_action_integral, vec)
        dyn_eqn = full_simplify(dyn_eqn)
        eom[str(vec)] = (vec, dyn_eqn)
    return eom
