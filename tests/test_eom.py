"""Tests for geomech.dynamics — equations of motion via least action."""

from geomech.core.base.expressions import Scalar, Vector, Matrix, S2, SO3
from geomech.core.operations.geometry import Dot
from geomech.core.operations.calculus import Variation
from geomech.dynamics import SystemVariables, compute_eom
from geomech.utils.printing import print_eom, print_tree


def _getScalars(names, attr=None):
    return [Scalar(n, attr=attr) for n in names.split()]


# ===========================================================================
# Point mass: L = 0.5 m v·v - m g x·e3,  δW = δx·f
# ===========================================================================

class TestPointMassEOM:
    """Point mass in 3D — simplest possible EOM test."""

    def test_produces_eom_dict(self):
        m, g = _getScalars('m g', attr=['Constant'])
        e3 = Vector('e3', attr=['Constant'])
        x = Vector('x')
        f = Vector('f')

        v = x.t_diff()

        PE = m * Dot(x, g * e3)
        KE = m * Dot(v, v) * Scalar('0.5', value=0.5, attr=['Constant'])
        L = KE - PE

        delta_x = x.get_variation_vector()  # Variation(x)
        deltaW = Dot(delta_x, f)

        variables = SystemVariables(vectors=[x])
        eqs = compute_eom(L, deltaW, variables)
        assert isinstance(eqs, dict)
        assert len(eqs) == 1

        print('\n--- Point Mass: Lagrangian tree ---')
        print_tree(L)
        print('\n--- Point Mass: EOM ---')
        print_eom(eqs)
        print('\n--- Point Mass: EOM tree ---')
        for _, (_, eqn) in eqs.items():
            print_tree(eqn)

    def test_variation_vector_is_key(self):
        m, = _getScalars('m', attr=['Constant'])
        x = Vector('x')
        f = Vector('f')

        v = x.t_diff()
        L = m * Dot(v, v) * Scalar('0.5', value=0.5, attr=['Constant'])
        deltaW = Dot(x.get_variation_vector(), f)

        variables = SystemVariables(vectors=[x])
        eqs = compute_eom(L, deltaW, variables)

        # key should be the string form of Variation(x)
        assert str(Variation(x)) in eqs

    def test_eom_value_is_tuple(self):
        m, = _getScalars('m', attr=['Constant'])
        x = Vector('x')
        f = Vector('f')

        v = x.t_diff()
        L = m * Dot(v, v) * Scalar('0.5', value=0.5, attr=['Constant'])
        deltaW = Dot(x.get_variation_vector(), f)

        variables = SystemVariables(vectors=[x])
        eqs = compute_eom(L, deltaW, variables)

        key = str(Variation(x))
        vec, eqn = eqs[key]
        assert vec == Variation(x)
        # equation should be a vector expression (the EOM)
        assert eqn is not None


# ===========================================================================
# Spherical pendulum (S2 manifold)
# ===========================================================================

class TestSphericalPendulumEOM:
    def test_spherical_pendulum_produces_eom(self):
        m, g, l = _getScalars('m g l', attr=['Constant'])
        e3 = Vector('e3', attr=['Constant'])
        q = S2('q')
        f = Vector('f')

        x = l * q
        v = x.t_diff()

        PE = m * Dot(x, g * e3)
        KE = m * Dot(v, v) * Scalar('0.5', value=0.5, attr=['Constant'])
        L = KE - PE
        dW = Dot(q.get_variation_vector(), f)

        variables = SystemVariables(vectors=[q])
        eqs = compute_eom(L, dW, variables)

        assert isinstance(eqs, dict)
        assert len(eqs) == 1

        print('\n--- Spherical Pendulum: Lagrangian tree ---')
        print_tree(L)
        print('\n--- Spherical Pendulum: EOM ---')
        print_eom(eqs)
        print('\n--- Spherical Pendulum: EOM tree ---')
        for _, (_, eqn) in eqs.items():
            print_tree(eqn)


# ===========================================================================
# Rigid pendulum (SO3 manifold)
# ===========================================================================

class TestRigidPendulumEOM:
    def test_rigid_pendulum_produces_eom(self):
        J = Matrix('J', attr=['Constant', 'SymmetricMatrix'])
        rho = Vector('\\rho', attr=['Constant'])
        m, g = _getScalars('m g', attr=['Constant'])
        e3 = Vector('e3', attr=['Constant'])
        M_torque = Vector('M')
        R = SO3('R')
        Om = R.get_tangent_vector()
        eta = R.get_variation_vector()

        x = R * rho
        v = x.t_diff()

        KE = Dot(Om, J * Om) * Scalar('0.5', value=0.5, attr=['Constant']) \
           + Dot(v, v) * m * Scalar('0.5', value=0.5, attr=['Constant'])
        PE = m * g * Dot(x, e3)
        L = KE - PE
        deltaW = Dot(eta, M_torque)

        variables = SystemVariables(matrices=[R])
        eqs = compute_eom(L, deltaW, variables)

        assert isinstance(eqs, dict)
        assert len(eqs) == 1

        print('\n--- Rigid Pendulum: Lagrangian tree ---')
        print_tree(L)
        print('\n--- Rigid Pendulum: EOM ---')
        print_eom(eqs)
        print('\n--- Rigid Pendulum: EOM tree ---')
        for _, (_, eqn) in eqs.items():
            print_tree(eqn)


# ===========================================================================
# SystemVariables dataclass
# ===========================================================================

class TestSystemVariables:
    def test_defaults_are_empty(self):
        sv = SystemVariables()
        assert sv.scalars == []
        assert sv.vectors == []
        assert sv.matrices == []

    def test_accepts_vectors(self):
        x = Vector('x')
        sv = SystemVariables(vectors=[x])
        assert sv.vectors == [x]

    def test_accepts_matrices(self):
        R = SO3('R')
        sv = SystemVariables(matrices=[R])
        assert sv.matrices == [R]

    def test_accepts_mixed(self):
        s = Scalar('s')
        x = Vector('x')
        R = SO3('R')
        sv = SystemVariables(scalars=[s], vectors=[x], matrices=[R])
        assert len(sv.scalars) == 1
        assert len(sv.vectors) == 1
        assert len(sv.matrices) == 1
