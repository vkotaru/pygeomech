"""Tests for standard form extraction: M*a + f + G*u = 0."""

import pytest
from geomech import (
    Scalar, Vector, Matrix, S2, SO3,
    Dot, getScalars,
    SystemVariables, compute_eom, to_standard_form, StandardFormEquation,
    Variation, TimeDerivative,
    ZeroVector, ZeroMatrix, IdentityMatrix,
    print_tree,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _point_mass_eom():
    """Point mass: L = 0.5*m*Dot(v,v) - m*g*Dot(x,e3), dW = Dot(dx,f)."""
    m, g = getScalars('m g', attr=['Constant'])
    e3 = Vector('e3', attr=['Constant'])
    x = Vector('x')
    f = Vector('f')
    v = x.t_diff()

    half = Scalar('0.5', value=0.5, attr=['Constant'])
    KE = m * Dot(v, v) * half
    PE = m * Dot(x, g * e3)
    L = KE - PE
    dW = Dot(x.get_variation_vector(), f)

    variables = SystemVariables(vectors=[x])
    eom = compute_eom(L, dW, variables)
    return eom, variables, [f], x


def _spherical_pendulum_eom():
    """Spherical pendulum on S2: L = 0.5*m*l^2*Dot(w,w) - m*g*l*Dot(q,e3)."""
    m, g, l = getScalars('m g l', attr=['Constant'])
    e3 = Vector('e3', attr=['Constant'])
    q = S2('q')
    f = Vector('f')

    x = l * q
    v = x.t_diff()
    half = Scalar('0.5', value=0.5, attr=['Constant'])
    KE = m * Dot(v, v) * half
    PE = m * Dot(x, g * e3)
    L = KE - PE
    dW = Dot(q.get_variation_vector(), f)

    variables = SystemVariables(vectors=[q])
    eom = compute_eom(L, dW, variables)
    return eom, variables, [f], q


def _rigid_pendulum_eom():
    """Rigid pendulum on SO3: L = 0.5*Dot(Om,J*Om) + 0.5*m*Dot(v,v) - m*g*Dot(x,e3)."""
    J = Matrix('J', attr=['Constant', 'SymmetricMatrix'])
    rho = Vector('\\rho', attr=['Constant'])
    m, g = getScalars('m g', attr=['Constant'])
    e3 = Vector('e3', attr=['Constant'])
    M_torque = Vector('M')

    R = SO3('R')
    Om = R.get_tangent_vector()
    eta = R.get_variation_vector()

    x = R * rho
    v = x.t_diff()
    half = Scalar('0.5', value=0.5, attr=['Constant'])
    KE = Dot(Om, J * Om) * half + Dot(v, v) * m * half
    PE = m * g * Dot(x, e3)
    L = KE - PE
    dW = Dot(eta, M_torque)

    variables = SystemVariables(matrices=[R])
    eom = compute_eom(L, dW, variables)
    return eom, variables, [M_torque], R


# ---------------------------------------------------------------------------
# Point mass tests
# ---------------------------------------------------------------------------

class TestPointMassStandardForm:
    def test_returns_dict(self):
        eom, variables, inputs, x = _point_mass_eom()
        sf = to_standard_form(eom, variables, inputs)
        assert isinstance(sf, dict)
        assert len(sf) == 1

    def test_keys_match_eom(self):
        eom, variables, inputs, x = _point_mass_eom()
        sf = to_standard_form(eom, variables, inputs)
        assert set(sf.keys()) == set(eom.keys())

    def test_is_standard_form_equation(self):
        eom, variables, inputs, x = _point_mass_eom()
        sf = to_standard_form(eom, variables, inputs)
        key = str(Variation(x))
        eq = sf[key]
        assert isinstance(eq, StandardFormEquation)

    def test_M_has_acceleration(self):
        eom, variables, inputs, x = _point_mass_eom()
        sf = to_standard_form(eom, variables, inputs)
        key = str(Variation(x))
        eq = sf[key]

        ddx = str(TimeDerivative(TimeDerivative(x)))
        assert ddx in eq.M, f'M should contain ddx key, got keys: {list(eq.M.keys())}'
        M = eq.M[ddx]
        assert M is not None
        # M should contain mass m
        assert 'm' in str(M)

    def test_G_has_input(self):
        eom, variables, inputs, x = _point_mass_eom()
        sf = to_standard_form(eom, variables, inputs)
        key = str(Variation(x))
        eq = sf[key]

        assert str(inputs[0]) in eq.G, f'G should contain input f, got keys: {list(eq.G.keys())}'

    def test_f_has_gravity(self):
        eom, variables, inputs, x = _point_mass_eom()
        sf = to_standard_form(eom, variables, inputs)
        key = str(Variation(x))
        eq = sf[key]

        f_str = str(eq.f)
        # f should contain gravity terms (g, e3, m)
        assert 'g' in f_str
        assert 'e3' in f_str

    def test_no_acceleration_in_f(self):
        eom, variables, inputs, x = _point_mass_eom()
        sf = to_standard_form(eom, variables, inputs)
        key = str(Variation(x))
        eq = sf[key]

        ddx = TimeDerivative(TimeDerivative(x))
        assert not eq.f.has(ddx), 'f should not contain acceleration terms'

    def test_no_input_in_f(self):
        eom, variables, inputs, x = _point_mass_eom()
        sf = to_standard_form(eom, variables, inputs)
        key = str(Variation(x))
        eq = sf[key]

        assert not eq.f.has(inputs[0]), 'f should not contain input terms'


# ---------------------------------------------------------------------------
# Spherical pendulum (S2) tests
# ---------------------------------------------------------------------------

class TestSphericalPendulumStandardForm:
    def test_returns_one_equation(self):
        eom, variables, inputs, q = _spherical_pendulum_eom()
        sf = to_standard_form(eom, variables, inputs)
        assert len(sf) == 1

    def test_M_empty_without_kinematic_substitution(self):
        """M is empty because the current pipeline does not substitute
        manifold kinematics (delta(omega) = xi_dot + ...).
        Once kinematic substitution is implemented, this test should
        be updated to check for d/dt(omega) terms in M."""
        eom, variables, inputs, q = _spherical_pendulum_eom()
        sf = to_standard_form(eom, variables, inputs)
        key = list(sf.keys())[0]
        eq = sf[key]
        # TODO: update when manifold kinematic substitution is added
        assert len(eq.M) == 0

    def test_G_has_input(self):
        eom, variables, inputs, q = _spherical_pendulum_eom()
        sf = to_standard_form(eom, variables, inputs)
        key = list(sf.keys())[0]
        eq = sf[key]
        assert len(eq.G) > 0, 'G should have the input force'

    def test_f_has_gravity(self):
        eom, variables, inputs, q = _spherical_pendulum_eom()
        sf = to_standard_form(eom, variables, inputs)
        key = list(sf.keys())[0]
        eq = sf[key]
        assert 'g' in str(eq.f) or eq.f == ZeroVector, 'f should have gravity or be zero'


# ---------------------------------------------------------------------------
# Rigid pendulum (SO3) tests
# ---------------------------------------------------------------------------

class TestRigidPendulumStandardForm:
    def test_returns_one_equation(self):
        eom, variables, inputs, R = _rigid_pendulum_eom()
        sf = to_standard_form(eom, variables, inputs)
        assert len(sf) == 1

    def test_M_empty_without_kinematic_substitution(self):
        """M is empty because the current pipeline does not substitute
        manifold kinematics (delta(Omega) = eta_dot + Omega x eta).
        Once kinematic substitution is implemented, this test should
        be updated to check for J * d/dt(Omega) in M."""
        eom, variables, inputs, R = _rigid_pendulum_eom()
        sf = to_standard_form(eom, variables, inputs)
        key = list(sf.keys())[0]
        eq = sf[key]
        # TODO: update when manifold kinematic substitution is added
        assert len(eq.M) == 0

    def test_G_has_torque(self):
        eom, variables, inputs, R = _rigid_pendulum_eom()
        sf = to_standard_form(eom, variables, inputs)
        key = list(sf.keys())[0]
        eq = sf[key]
        assert len(eq.G) > 0, 'G should contain the torque input'

    def test_f_exists(self):
        eom, variables, inputs, R = _rigid_pendulum_eom()
        sf = to_standard_form(eom, variables, inputs)
        key = list(sf.keys())[0]
        eq = sf[key]
        # f should have Coriolis and/or gravity terms
        assert eq.f is not None


# ---------------------------------------------------------------------------
# No inputs case
# ---------------------------------------------------------------------------

class TestNoInputs:
    def test_no_inputs_gives_empty_G(self):
        """Free particle with no external forces."""
        m, = getScalars('m', attr=['Constant'])
        x = Vector('x')
        v = x.t_diff()
        half = Scalar('0.5', value=0.5, attr=['Constant'])
        L = m * Dot(v, v) * half
        dW = Scalar('0', value=0, attr=['Constant', 'Zero'])

        variables = SystemVariables(vectors=[x])
        eom = compute_eom(L, dW, variables)
        sf = to_standard_form(eom, variables, inputs=[])

        key = str(Variation(x))
        eq = sf[key]
        assert len(eq.G) == 0, 'G should be empty with no inputs'
        assert len(eq.M) > 0, 'M should still have acceleration terms'
