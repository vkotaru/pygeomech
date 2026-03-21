"""End-to-end tests for manifold EOM and standard form pipelines.

Tests the full pipeline: Lagrangian → variation → manifold rules →
simplify → IBP → expand → extract → standard form for S2, SO3,
and mixed systems.
"""

import pytest
from geomech import (
    Scalar, Vector, Matrix, S2, SO3,
    Dot, Cross, Hat, getScalars,
    SystemVariables, compute_eom, to_standard_form, StandardFormEquation,
    Variation, TimeDerivative,
    ZeroVector, ZeroMatrix, IdentityMatrix,
    full_simplify, expand,
)
from geomech.core.operations.multiplication import MVMul


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _spherical_pendulum():
    """Spherical pendulum on S2."""
    m, g, l = getScalars('m g l', attr=['Constant'])
    e3 = Vector('e3', attr=['Constant'])
    q = S2('q')
    f = Vector('f')
    omega = q.get_tangent_vector()
    xi = q.get_variation_vector()

    half = Scalar('0.5', value=0.5, attr=['Constant'])
    KE = half * m * l * l * Dot(omega, omega)
    PE = m * g * l * Dot(q, e3)
    L = KE - PE
    dW = Dot(xi, f)

    variables = SystemVariables(vectors=[q])
    eom = compute_eom(L, dW, variables)
    return eom, variables, [f], q


def _rigid_body():
    """Free rigid body on SO3 (no gravity, no offset)."""
    J = Matrix('J', attr=['Constant', 'SymmetricMatrix'])
    M_torque = Vector('M')
    R = SO3('R')
    Om = R.get_tangent_vector()
    eta = R.get_variation_vector()

    half = Scalar('0.5', value=0.5, attr=['Constant'])
    L = Dot(Om, J * Om) * half
    dW = Dot(eta, M_torque)

    variables = SystemVariables(matrices=[R])
    eom = compute_eom(L, dW, variables)
    return eom, variables, [M_torque], R


def _rigid_pendulum():
    """Rigid pendulum on SO3 with gravity."""
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
# S2 spherical pendulum
# ---------------------------------------------------------------------------

class TestSphericalPendulumPipeline:
    def test_eom_produces_one_equation(self):
        eom, _, _, q = _spherical_pendulum()
        assert len(eom) == 1

    def test_eom_key_is_xi(self):
        eom, _, _, q = _spherical_pendulum()
        xi = q.get_variation_vector()
        assert str(xi) in eom

    def test_eom_contains_acceleration(self):
        """EOM should contain d/dt(omega)."""
        eom, _, _, q = _spherical_pendulum()
        omega = q.get_tangent_vector()
        key = list(eom.keys())[0]
        _, eqn = eom[key]
        assert eqn.has(TimeDerivative(omega))

    def test_eom_contains_gravity(self):
        eom, _, _, q = _spherical_pendulum()
        key = list(eom.keys())[0]
        _, eqn = eom[key]
        assert 'e3' in str(eqn) or 'g' in str(eqn)

    def test_standard_form_M_has_inertia(self):
        eom, variables, inputs, q = _spherical_pendulum()
        sf = to_standard_form(eom, variables, inputs)
        key = list(sf.keys())[0]
        eq = sf[key]
        omega = q.get_tangent_vector()
        ddw = str(TimeDerivative(omega))
        assert ddw in eq.M
        # M should contain mass and length terms
        M_str = str(eq.M[ddw])
        assert 'm' in M_str
        assert 'l' in M_str

    def test_standard_form_G_has_input(self):
        eom, variables, inputs, q = _spherical_pendulum()
        sf = to_standard_form(eom, variables, inputs)
        key = list(sf.keys())[0]
        eq = sf[key]
        assert 'f' in eq.G

    def test_standard_form_f_has_gravity(self):
        eom, variables, inputs, q = _spherical_pendulum()
        sf = to_standard_form(eom, variables, inputs)
        key = list(sf.keys())[0]
        eq = sf[key]
        f_str = str(eq.f)
        assert 'e3' in f_str or 'g' in f_str

    def test_no_eta_dot_in_eom(self):
        """After IBP, no d/dt(xi) should remain."""
        eom, _, _, q = _spherical_pendulum()
        xi = q.get_variation_vector()
        xi_dot = TimeDerivative(xi)
        key = list(eom.keys())[0]
        _, eqn = eom[key]
        assert not eqn.has(xi_dot), 'IBP should have removed d/dt(xi)'


# ---------------------------------------------------------------------------
# S2 Lagrangian formulation equivalence
# ---------------------------------------------------------------------------

class TestS2FormulationEquivalence:
    """Both formulations of S2 KE should produce the same EOM."""

    def test_omega_vs_qdot_formulation(self):
        """L written with Dot(omega,omega) vs Dot(v,v) where v=l*q_dot."""
        m, g, l = getScalars('m g l', attr=['Constant'])
        e3 = Vector('e3', attr=['Constant'])
        q = S2('q')
        f = Vector('f')
        omega = q.get_tangent_vector()
        xi = q.get_variation_vector()
        half = Scalar('0.5', value=0.5, attr=['Constant'])
        PE = m * g * l * Dot(q, e3)
        dW = Dot(xi, f)
        variables = SystemVariables(vectors=[q])

        # Version A: KE = 0.5*m*l^2*Dot(omega, omega)
        KE_A = half * m * l * l * Dot(omega, omega)
        eom_A = compute_eom(KE_A - PE, dW, variables)
        _, eqn_A = eom_A[str(xi)]

        # Version B: KE = 0.5*m*Dot(v, v) with v = (l*q).t_diff()
        x = l * q
        v = x.t_diff()
        KE_B = m * Dot(v, v) * half
        eom_B = compute_eom(KE_B - PE, dW, variables)
        _, eqn_B = eom_B[str(xi)]

        # Both should produce the same number of terms
        terms_A = len(eqn_A.nodes) if hasattr(eqn_A, 'nodes') else 1
        terms_B = len(eqn_B.nodes) if hasattr(eqn_B, 'nodes') else 1
        assert terms_A == terms_B, (
            f'Formulation mismatch: A has {terms_A} terms, B has {terms_B}. '
            f'A={eqn_A}, B={eqn_B}'
        )

    def test_qdot_has_no_spurious_coriolis(self):
        """v=l*q_dot formulation should not have extra Coriolis-like terms."""
        m, g, l = getScalars('m g l', attr=['Constant'])
        e3 = Vector('e3', attr=['Constant'])
        q = S2('q')
        f = Vector('f')
        xi = q.get_variation_vector()
        omega = q.get_tangent_vector()
        half = Scalar('0.5', value=0.5, attr=['Constant'])

        x = l * q
        v = x.t_diff()
        KE = m * Dot(v, v) * half
        PE = m * g * l * Dot(q, e3)
        L = KE - PE
        dW = Dot(xi, f)

        variables = SystemVariables(vectors=[q])
        eom = compute_eom(L, dW, variables)
        _, eqn = eom[str(xi)]

        # EOM should only have: omega_dot term, gravity term, input term
        # No Coriolis terms for a particle on S2 (centripetal cancels)
        omega_dot = TimeDerivative(omega)
        for n in (eqn.nodes if hasattr(eqn, 'nodes') else [eqn]):
            if n.has(omega) and not n.has(omega_dot) and 'e3' not in str(n) and str(n) != 'f':
                assert False, f'Unexpected Coriolis-like term: {n}'


# ---------------------------------------------------------------------------
# SO3 free rigid body
# ---------------------------------------------------------------------------

class TestFreeRigidBodyPipeline:
    def test_eom_produces_one_equation(self):
        eom, _, _, R = _rigid_body()
        assert len(eom) == 1

    def test_eom_key_is_eta(self):
        eom, _, _, R = _rigid_body()
        eta = R.get_variation_vector()
        assert str(eta) in eom

    def test_eom_contains_acceleration(self):
        eom, _, _, R = _rigid_body()
        Om = R.get_tangent_vector()
        key = list(eom.keys())[0]
        _, eqn = eom[key]
        assert eqn.has(TimeDerivative(Om))

    def test_standard_form_M_is_J(self):
        """For a free rigid body, M should be proportional to J."""
        eom, variables, inputs, R = _rigid_body()
        sf = to_standard_form(eom, variables, inputs)
        key = list(sf.keys())[0]
        eq = sf[key]
        Om = R.get_tangent_vector()
        ddOm = str(TimeDerivative(Om))
        assert ddOm in eq.M
        assert 'J' in str(eq.M[ddOm])

    def test_standard_form_G_has_torque(self):
        eom, variables, inputs, R = _rigid_body()
        sf = to_standard_form(eom, variables, inputs)
        key = list(sf.keys())[0]
        eq = sf[key]
        assert 'M' in eq.G

    def test_standard_form_f_has_coriolis(self):
        """Free rigid body should have Coriolis terms (Omega × J*Omega)."""
        eom, variables, inputs, R = _rigid_body()
        sf = to_standard_form(eom, variables, inputs)
        key = list(sf.keys())[0]
        eq = sf[key]
        Om = R.get_tangent_vector()
        f_str = str(eq.f)
        # f should contain Omega and J
        assert str(Om) in f_str or 'J' in f_str

    def test_no_eta_dot_in_eom(self):
        """After IBP, no d/dt(eta) should remain."""
        eom, _, _, R = _rigid_body()
        eta = R.get_variation_vector()
        eta_dot = TimeDerivative(eta)
        key = list(eom.keys())[0]
        _, eqn = eom[key]
        assert not eqn.has(eta_dot), 'IBP should have removed d/dt(eta)'


# ---------------------------------------------------------------------------
# SO3 rigid pendulum (with translational KE and gravity)
# ---------------------------------------------------------------------------

class TestRigidPendulumPipeline:
    def test_eom_produces_one_equation(self):
        eom, _, _, R = _rigid_pendulum()
        assert len(eom) == 1

    def test_standard_form_M_has_inertia(self):
        eom, variables, inputs, R = _rigid_pendulum()
        sf = to_standard_form(eom, variables, inputs)
        key = list(sf.keys())[0]
        eq = sf[key]
        Om = R.get_tangent_vector()
        ddOm = str(TimeDerivative(Om))
        assert ddOm in eq.M
        M_str = str(eq.M[ddOm])
        # Should contain J and rho (from translational inertia)
        assert 'J' in M_str

    def test_standard_form_f_has_gravity(self):
        eom, variables, inputs, R = _rigid_pendulum()
        sf = to_standard_form(eom, variables, inputs)
        key = list(sf.keys())[0]
        eq = sf[key]
        f_str = str(eq.f)
        # Should contain gravity terms
        assert 'e3' in f_str or 'g' in f_str

    def test_standard_form_G_has_torque(self):
        eom, variables, inputs, R = _rigid_pendulum()
        sf = to_standard_form(eom, variables, inputs)
        key = list(sf.keys())[0]
        eq = sf[key]
        assert 'M' in eq.G

    def test_standard_form_is_complete(self):
        """All three components M, f, G should be non-trivial."""
        eom, variables, inputs, R = _rigid_pendulum()
        sf = to_standard_form(eom, variables, inputs)
        key = list(sf.keys())[0]
        eq = sf[key]
        assert len(eq.M) > 0, 'M should have acceleration terms'
        assert eq.f != ZeroVector, 'f should have nonlinear terms'
        assert len(eq.G) > 0, 'G should have input terms'


# ---------------------------------------------------------------------------
# Simplification rules
# ---------------------------------------------------------------------------

class TestHatCrossSimplification:
    def test_hat_v_times_w_becomes_cross(self):
        """MVMul(Hat(v), w) should simplify to Cross(v, w)."""
        v = Vector('v')
        w = Vector('w')
        expr = MVMul(Hat(v), w)
        result = full_simplify(expr)
        assert isinstance(result, Cross)
        assert str(result.left) == 'v'
        assert str(result.right) == 'w'

    def test_hat_distributes_over_vadd(self):
        """Hat(x + y) → Hat(x) + Hat(y) after expand."""
        from geomech.core.operations.addition import VAdd, MAdd
        x = Vector('x')
        y = Vector('y')
        expr = Hat(VAdd(x, y))
        result = expand(expr)
        assert isinstance(result, MAdd)
        assert len(result.nodes) == 2

    def test_mmmul_mvmul_associativity(self):
        """MVMul(MMMul(A, B), v) → MVMul(A, MVMul(B, v)) after simplify."""
        A = Matrix('A', attr=['Constant'])
        v = Vector('v')
        w = Vector('w')
        expr = MVMul(A, MVMul(Hat(v), w))
        result = full_simplify(expr)
        # Should become MVMul(A, Cross(v, w))
        assert isinstance(result, MVMul)
        assert isinstance(result.right, Cross)


# ---------------------------------------------------------------------------
# Collect enhancements
# ---------------------------------------------------------------------------

class TestCollectManifoldPatterns:
    def test_cross_cross_collect(self):
        """Dot(Cross(vec, a), Cross(b, c)) should collect vec to left."""
        from geomech.core.math.collect import collect
        vec = Vector('vec')
        a = Vector('a', attr=['Constant'])
        b = Vector('b', attr=['Constant'])
        c = Vector('c', attr=['Constant'])
        expr = Dot(Cross(vec, a), Cross(b, c))
        result = collect(expr, vec)
        assert str(result.left) == 'vec'

    def test_so3_rotation_stripping(self):
        """Dot(R*a, R*b) → Dot(a, b) when R is SO3."""
        from geomech.core.math.collect import collect
        R = SO3('R')
        vec = Vector('vec')
        w = Vector('w', attr=['Constant'])
        expr = Dot(MVMul(R, vec), MVMul(R, w))
        result = collect(expr, vec)
        assert str(result.left) == 'vec'

    def test_mvmul_cross_deep_collect(self):
        """Dot(MVMul(R, Cross(vec, rho)), w) should collect vec."""
        from geomech.core.math.collect import collect
        from geomech.core.operations.geometry import Transpose
        R = SO3('R')
        vec = Vector('vec')
        rho = Vector('rho', attr=['Constant'])
        w = Vector('w', attr=['Constant'])
        expr = Dot(MVMul(R, Cross(vec, rho)), w)
        result = collect(expr, vec)
        assert str(result.left) == 'vec'
