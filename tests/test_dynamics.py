import pytest
import numpy as np
from geomech.base.scalars import Scalar, getScalars
from geomech.base.vectors import Vector, S2, getVectors
from geomech.base.matrices import Matrix, SO3, getMatrices
from geomech.operations.geometry import Dot
from geomech.operations.dynamics import compute_eom


class TestPointMassEOM:
    @pytest.mark.xfail(reason="expand() hits UndefinedCaseError on Delta nodes — known gap")
    def test_point_mass_produces_eom(self):
        m, g = getScalars('m g', attr=['Constant'])
        e3 = Vector('e3', attr=['Constant'], value=np.array([0., 0., 1]))
        x = Vector('x')
        f = Vector('f')
        v = x.diff()

        PE = m * x.dot(g * e3)
        KE = m * Dot(v, v) * 0.5
        L = KE - PE
        deltaW = Dot(x.delta(), f)

        eqs = compute_eom(L, deltaW, [[], [x], []])
        assert isinstance(eqs, dict)
        assert len(eqs) == 1


class TestSphericalPendulumEOM:
    @pytest.mark.xfail(reason="vector_rules() hits NotImplementedError on Dot(Cross,...) — known gap")
    def test_spherical_pendulum_produces_eom(self):
        m, g, l = getScalars('m g l', attr=['Constant'])
        e3 = Vector('e3', attr=['Constant'])
        q = S2('q')
        f = Vector('f')
        om = q.get_tangent_vector()

        x = l * q
        v = x.diff()

        PE = m * x.dot(g * e3)
        KE = m * Dot(v, v) * 0.5
        L = KE - PE
        dW = Dot(q.delta(), f)

        eqs = compute_eom(L, dW, [[], [q], []])
        assert isinstance(eqs, dict)
        assert len(eqs) == 1


class TestRigidPendulumEOM:
    @pytest.mark.xfail(reason="full_simplify() pipeline hits errors on rigid body terms — known gap")
    def test_rigid_pendulum_produces_eom(self):
        J = Matrix('J', attr=['Constant', 'SymmetricMatrix'])
        rho = Vector('\\rho', attr=['Constant'])
        m, g = getScalars('m g', attr=['Constant'])
        e3 = Vector('e3', attr=['Constant'])
        M_torque = Vector('M')
        R = SO3('R')
        Om = R.get_tangent_vector()
        eta = R.get_variation_vector()

        x = R * rho
        v = x.diff()

        KE = Dot(Om, J * Om) * 0.5 + Dot(v, v) * m * 0.5
        PE = m * g * Dot(x, e3)
        L = KE - PE
        deltaW = Dot(eta, M_torque)

        eqs = compute_eom(L, deltaW, [[], [], [R]])
        assert isinstance(eqs, dict)
        assert len(eqs) == 1
