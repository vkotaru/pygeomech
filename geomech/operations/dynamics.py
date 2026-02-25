from geomech.operations.algebraic_manipulation import extract_coeff
from geomech.operations.expansion import expand
from geomech.operations.integration import integrate_by_parts_vectors
from geomech.operations.simplification import full_simplify


def compute_eom(lagrangian, inf_work, variables):
    """
    Computes Lagrange-Hamiltonian dynamics using principle of least action
    """

    # taking variation of lagrangian
    dL = lagrangian.delta()
    # infinitesimal action integral

    dS = dL + inf_work
    dS = full_simplify(dS)

    # gather the variations (vectors)
    variation_vectors = []
    variation_vector_dots = []
    for vec in variables[1]:
        x = vec.get_variation_vector()
        variation_vectors.append(x)
        variation_vector_dots.append(x.diff())

    for mat in variables[2]:
        x = mat.get_variation_vector()
        variation_vectors.append(x)
        variation_vector_dots.append(x.diff())

    dS = integrate_by_parts_vectors(dS, variation_vector_dots)

    # simplify Tree
    dS = expand(dS)
    # TODO applyConstraints
    # TODO expand
    # TODO full_simplify
    # TODO organize
    # TODO split
    # TODO cancelTerms
    # extract
    eom_dict = separate_variations(dS, variation_vectors)
    return eom_dict


def separate_variations(inf_action_integral, variation_vectors):
    _dict = {}

    for vec in variation_vectors:
        dyn_eqn = extract_coeff(inf_action_integral, vec)
        dyn_eqn = full_simplify(dyn_eqn)
        _dict[vec.__str__()] = (vec, dyn_eqn)

    return _dict
