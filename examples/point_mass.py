from geomech import *
from geomech.operations.print_tree import print_latex
from geomech.utils.errors import ExpressionMismatchError
import numpy as np


def point_mass():
    """
    Derives Newton's equation of motion for 3d point mass

    Variables
    scalars: mass: m [kg]
             gravity: g [m/s^2]

    vectors: neg gravity direction: e3 [0,0,1] [No units]
             point-mass position: x  [m]
             input force: f [N]

    :return: None
    """
    m, g = getScalars('m g', attr=['Constant'])
    e3 = Vector('e3', attr=['Constant'], value=np.array([0., 0., 1]))
    x, f = getVectors(['x', 'f'])

    v = x.diff()

    # computing energies
    PE = m * x.dot(g * e3)
    KE = m * Dot(v, v) * 0.5

    # Lagrangian
    L = KE - PE

    # infinitesimal work
    deltaW = Dot(x.delta(), f)

    eqs = compute_eom(L, deltaW, [[], [x], []])
    print_latex(eqs)

    print('done')


def two_point_masses():
    """two point masses"""
    m1, m2, g = getScalars('m1 m2 g', attr=['Constant'])
    e3 = Vector('e3', attr=['Constant'], value=np.array([0., 0., 1]))
    x1, f1, x2, f2 = getVectors(['x1', 'f1', 'x2', 'f2'])

    v1, v2 = x1.diff(), x2.diff()
    # computing energies
    PE = m1 * x1.dot((g * e3)) + m2 * x2.dot(g * e3)
    KE = m1 * Dot(v1, v1) * 0.5 + m2 * Dot(v2, v2) * 0.5

    # Lagrangian
    L = KE - PE
    # infinitesimal work
    dW = Dot(x1.delta(), f1) + Dot(x2.delta(), f2)

    eqs = compute_eom(L, dW, [[], [x1, x2], []])
    print_latex(eqs)

    print('done')


if __name__ == "__main__":
    two_point_masses()
    # point_mass()
