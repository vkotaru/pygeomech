from geomech import *
import numpy as np


def rigid_pendulum():
    J = Matrix('J', attr=['Constant', 'SymmetricMatrix'])
    rho = Vector('\\rho', attr=['Constant'])
    m, g = getScalars('m g', attr=['Constant'])
    e3 = Vector('e_3', attr=['Constant'], value=np.array([0., 0., 1]))

    M = Vector('M')
    R = SO3('R')
    Om = R.get_tangent_vector()
    eta = R.get_variation_vector()
    x = R * rho
    v = x.t_diff()

    KE = Dot(Om, J * Om) * 0.5 + Dot(v, v) * m * 0.5
    PE = m * g * Dot(x, e3)
    L = KE - PE

    deltaW = Dot(eta, M)
    eqns = compute_eom(L, deltaW, SystemVariables(matrices=[R]))
    print_eom(eqns)

    print('done')


def double_rigid_pendulum():
    """double rigid pendulum"""
    J1 = Matrix('J1', attr=['Constant', 'SymmetricMatrix'])
    J2 = Matrix('J1', attr=['Constant', 'SymmetricMatrix'])
    rho1 = Vector('\\rho1', attr=['Constant'])
    rho2 = Vector('\\rho2', attr=['Constant'])
    l1 = Vector('l1', attr=['Constant'])
    m1, m2, g = getScalars('m1 m2 g', attr=['Constant'])
    e3 = Vector('e_3', attr=['Constant'], value=np.array([0., 0., 1]))

    M1, M2 = getVectors('M1 M2')
    R1, R2 = SO3('R1'), SO3('R2')
    Om1, eta1 = R1.get_tangent_vector(), R1.get_variation_vector()
    Om2, eta2 = R2.get_tangent_vector(), R2.get_variation_vector()

    x1 = R1 * rho1
    x2 = R1 * l1 + R2 * rho2
    v2 = x2.t_diff()

    KE = Dot(Om1, J1 * Om1) * 0.5 + Dot(Om2, J2 * Om2) * 0.5 + Dot(v2, v2) * m2 * 0.5
    PE = (m1 * g * Dot(R1 * rho1, e3)) + (m2 * g * Dot(R2 * rho2, e3))
    L = KE - PE

    deltaW = Dot(eta1, M1) + Dot(eta2, M2)

    eqns = compute_eom(L, deltaW, SystemVariables(matrices=[R1, R2]))
    print_eom(eqns)

    print('done')


if __name__ == "__main__":
    # rigid_pendulum()
    double_rigid_pendulum()
