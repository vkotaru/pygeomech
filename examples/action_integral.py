from geomech import *
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
    """
    m, g = getScalars('m g', attr=['Constant'])
    e3 = Vector('e3', attr=['Constant'], value=np.array([0., 0., 1]))
    x, f = getVectors(['x', 'f'])

    v = x.t_diff()

    # computing energies
    PE = m * x.dot(g * e3)
    KE = m * Dot(v, v) * 0.5

    # Lagrangian
    L = KE - PE

    # infinitesimal work
    deltaW = Dot(x.delta(), f)

    eqs = compute_eom(L, deltaW, SystemVariables(vectors=[x]))
    print_eom(eqs)

    print('done')


def two_point_masses():
    """
    two point masses
    """
    m1, m2, g = getScalars('m1 m2 g', attr=['Constant'])
    e3 = Vector('e3', attr=['Constant'], value=np.array([0., 0., 1]))
    x1, f1, x2, f2 = getVectors(['x1', 'f1', 'x2', 'f2'])

    v1, v2 = x1.t_diff(), x2.t_diff()
    # computing energies
    PE = m1 * x1.dot((g * e3)) + m2 * x2.dot(g * e3)
    KE = m1 * Dot(v1, v1) * 0.5 + m2 * Dot(v2, v2) * 0.5

    # Lagrangian
    L = KE - PE
    # infinitesimal work
    dW = Dot(x1.delta(), f1) + Dot(x2.delta(), f2)

    eqs = compute_eom(L, dW, SystemVariables(vectors=[x1, x2]))
    print_eom(eqs)


def iteration_point_masses():
    """
    iteration_point_masses
    """
    # TODO
    pass


def spherical_pendulum():
    """
    Derives equation of motion for a spherical pendulum

    Variables
    scalars:    mass: m [kg]
                length: l [m]
                gravity: g [m/s^2]

    vectors: neg gravity direction: e3 [0,0,1] [No units]
             pendulum attitude: q in S2 [No units]
             input force: f [N]
    """
    # define constant scalars
    # mass, acceleration due to gravity, length of the pendulum
    m, g, l = getScalars('m g l', attr=['Constant'])

    e3 = Vector('e3', attr=['Constant'], value=np.array([0., 0., 1]))  # gravity direction

    q = S2('q')
    om = q.get_tangent_vector()
    f = Vector('f')  # external force

    x = l * q
    v = x.t_diff()
    # computing energies
    PE = m * x.dot((g * e3))
    KE = m * Dot(v, v) * 0.5

    # Lagrangian
    L = KE - PE
    # infinitesimal work
    dW = Dot(q.delta(), f)

    eqs = compute_eom(L, dW, SystemVariables(vectors=[q]))
    print_eom(eqs)


if __name__ == "__main__":
    # print('\n-----------------------------------\n')
    # print('generating dynamics for two independent point masses')
    # two_point_masses()

    print('\n-----------------------------------\n')
    print('generating dynamics for spherical pendulum')
    spherical_pendulum()
