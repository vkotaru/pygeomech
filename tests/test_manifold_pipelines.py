"""End-to-end tests for manifold EOM and standard form pipelines.

Tests the full pipeline: Lagrangian → variation → manifold rules →
simplify → IBP → expand → extract → standard form for S2, SO3,
and mixed systems.

Each test class verifies the actual generated dynamics against the
known analytical result from Lee, Leok, McClamroch (2018).
"""

from geomech import (
    S2,
    SO3,
    Dot,
    Matrix,
    Scalar,
    SystemVariables,
    TimeDerivative,
    Variation,
    Vector,
    compute_eom,
    getScalars,
    to_standard_form,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _spherical_pendulum():
    """Spherical pendulum on S2 — angular velocity formulation.

    KE = ½ m l² (ω · ω)  [modified Lagrangian L̃(q,ω), Lee et al. §5.3.3]
    """
    m, g, l = getScalars("m g l", attr=["Constant"])
    e3 = Vector("e3", attr=["Constant"])
    q = S2("q")
    f = Vector("f")
    omega = q.get_tangent_vector()
    xi = q.get_variation_vector()

    half = Scalar("0.5", value=0.5, attr=["Constant"])
    KE = half * m * l * l * Dot(omega, omega)
    PE = m * g * l * Dot(q, e3)
    L = KE - PE
    dW = Dot(xi, f)

    variables = SystemVariables(vectors=[q])
    eom = compute_eom(L, dW, variables)
    return eom, variables, [f], q


def _spherical_pendulum_qdot():
    """Spherical pendulum on S2 — configuration velocity formulation.

    x = l*q,  v = ẋ = l*(ω × q)
    KE = ½ m (v · v)  [standard Lagrangian L(q,q̇), Lee et al. §5.3.1]

    Internally v = l*(ω × q), so ‖v‖² = l²‖ω‖² (since ω ⊥ q, ‖q‖=1).
    This is the standard Lagrangian formulation from Lee et al. §5.3.1.
    """
    m, g, l = getScalars("m g l", attr=["Constant"])
    e3 = Vector("e3", attr=["Constant"])
    q = S2("q")
    f = Vector("f")
    xi = q.get_variation_vector()

    half = Scalar("0.5", value=0.5, attr=["Constant"])
    x = l * q
    v = x.t_diff()
    KE = m * Dot(v, v) * half
    PE = m * g * l * Dot(q, e3)
    L = KE - PE
    dW = Dot(xi, f)

    variables = SystemVariables(vectors=[q])
    eom = compute_eom(L, dW, variables)
    return eom, variables, [f], q


def _rigid_body():
    """Free rigid body on SO3 (no gravity, no offset)."""
    J = Matrix("J", attr=["Constant", "SymmetricMatrix"])
    M_torque = Vector("M")
    R = SO3("R")
    Om = R.get_tangent_vector()
    eta = R.get_variation_vector()

    half = Scalar("0.5", value=0.5, attr=["Constant"])
    L = Dot(Om, J * Om) * half
    dW = Dot(eta, M_torque)

    variables = SystemVariables(matrices=[R])
    eom = compute_eom(L, dW, variables)
    return eom, variables, [M_torque], R


def _rigid_pendulum_pivot():
    """Rigid pendulum on SO3 — J about pivot (Lee et al. Eq 6.8).

    J is the total inertia about the fixed pivot. KE = ½ Ω^T J Ω.
    PE = mg*(R*ρ)·e3 where ρ is pivot-to-COM in body frame.
    """
    J = Matrix("J", attr=["Constant", "SymmetricMatrix"])
    rho = Vector("\\rho", attr=["Constant"])
    m, g = getScalars("m g", attr=["Constant"])
    e3 = Vector("e3", attr=["Constant"])
    M_torque = Vector("M")

    R = SO3("R")
    Om = R.get_tangent_vector()
    eta = R.get_variation_vector()

    half = Scalar("0.5", value=0.5, attr=["Constant"])
    KE = Dot(Om, J * Om) * half
    PE = m * g * Dot(R * rho, e3)
    L = KE - PE
    dW = Dot(eta, M_torque)

    variables = SystemVariables(matrices=[R])
    eom = compute_eom(L, dW, variables)
    return eom, variables, [M_torque], R


def _rigid_pendulum_com():
    """Rigid pendulum on SO3 — J about COM (parallel axis theorem).

    J is the COM inertia. Total KE = ½ Ω^T J Ω + ½ m ‖v_COM‖²
    where v_COM = d/dt(R*ρ) = R*Hat(Ω)*ρ, so the translational term
    contributes m*hat(ρ)² to the effective inertia about the pivot.
    """
    J = Matrix("J", attr=["Constant", "SymmetricMatrix"])
    rho = Vector("\\rho", attr=["Constant"])
    m, g = getScalars("m g", attr=["Constant"])
    e3 = Vector("e3", attr=["Constant"])
    M_torque = Vector("M")

    R = SO3("R")
    Om = R.get_tangent_vector()
    eta = R.get_variation_vector()

    x = R * rho
    v = x.t_diff()
    half = Scalar("0.5", value=0.5, attr=["Constant"])
    KE = Dot(Om, J * Om) * half + Dot(v, v) * m * half
    PE = m * g * Dot(x, e3)
    L = KE - PE
    dW = Dot(eta, M_torque)

    variables = SystemVariables(matrices=[R])
    eom = compute_eom(L, dW, variables)
    return eom, variables, [M_torque], R


def _double_rigid_pendulum():
    """Double rigid pendulum on SO3 × SO3."""
    J1 = Matrix("J1", attr=["Constant", "SymmetricMatrix"])
    J2 = Matrix("J2", attr=["Constant", "SymmetricMatrix"])
    rho1 = Vector("\\rho_1", attr=["Constant"])
    rho2 = Vector("\\rho_2", attr=["Constant"])
    l1 = Vector("l_1", attr=["Constant"])
    m1, m2, g = getScalars("m1 m2 g", attr=["Constant"])
    e3 = Vector("e3", attr=["Constant"])
    M1 = Vector("M1")
    M2 = Vector("M2")

    R1 = SO3("R1")
    R2 = SO3("R2")
    Om1 = R1.get_tangent_vector()
    Om2 = R2.get_tangent_vector()
    eta1 = R1.get_variation_vector()
    eta2 = R2.get_variation_vector()

    half = Scalar("0.5", value=0.5, attr=["Constant"])

    x1 = R1 * rho1
    x2 = R1 * l1 + R2 * rho2
    v1 = x1.t_diff()
    v2 = x2.t_diff()

    KE = (
        Dot(Om1, J1 * Om1) * half
        + Dot(Om2, J2 * Om2) * half
        + Dot(v1, v1) * m1 * half
        + Dot(v2, v2) * m2 * half
    )
    PE = m1 * g * Dot(x1, e3) + m2 * g * Dot(x2, e3)
    L = KE - PE
    dW = Dot(eta1, M1) + Dot(eta2, M2)

    variables = SystemVariables(matrices=[R1, R2])
    eom = compute_eom(L, dW, variables)
    return eom, variables, [M1, M2], R1, R2


def _simple_se3_system():
    """Simple R3 × SO3 system (point mass + rigid body, no coupling)."""
    m = Scalar("m", attr=["Constant"])
    J = Matrix("J", attr=["Constant", "SymmetricMatrix"])
    half = Scalar("0.5", value=0.5, attr=["Constant"])
    F = Vector("F")
    M_torque = Vector("M")

    x = Vector("x")
    R = SO3("R")
    Om = R.get_tangent_vector()
    eta = R.get_variation_vector()

    v = x.t_diff()
    KE = m * Dot(v, v) * half + Dot(Om, J * Om) * half
    L = KE
    dW = Dot(x.delta(), F) + Dot(eta, M_torque)

    variables = SystemVariables(vectors=[x], matrices=[R])
    eom = compute_eom(L, dW, variables)
    return eom, variables, [F, M_torque], x, R


# ---------------------------------------------------------------------------
# S2 spherical pendulum — Lee et al. Eq 5.14
#
# Expected: ml²ω̇ + mgl(q × e3) = f
# Pipeline convention (all on one side): -ml²ω̇ - mgl(q×e3) + f = 0
# ---------------------------------------------------------------------------


class TestSphericalPendulumPipeline:
    def setup_method(self):
        self.eom, self.variables, self.inputs, self.q = _spherical_pendulum()
        self.omega = self.q.get_tangent_vector()
        self.xi = self.q.get_variation_vector()
        self.key = list(self.eom.keys())[0]
        _, self.eqn = self.eom[self.key]
        self.sf = to_standard_form(self.eom, self.variables, self.inputs)
        self.eq = self.sf[self.key]

    def test_eom_key_is_xi(self):
        assert str(self.xi) in self.eom

    def test_eom_str(self):
        """Full EOM string: d/dt(ω)*(-1)*m*l*l + Cross(q,e3)*m*g*l*(-1) + f"""
        expected = "\\frac{d}{dt}(\\omega_{q})(-1)mll+Cross(q,e3)mgl(-1)+f"
        assert str(self.eqn) == f"({expected})"

    def test_M(self):
        """M[ω̇] = I*(-1)*m*l*l  (= -ml² I)"""
        ddw = str(TimeDerivative(self.omega))
        assert ddw in self.eq.M
        assert str(self.eq.M[ddw]) == "I(-1)mll"

    def test_f(self):
        """f = Cross(q,e3)*m*g*l*(-1)  (= -mgl q×e3)"""
        assert str(self.eq.f) == "Cross(q,e3)mgl(-1)"

    def test_G(self):
        """G[f] = I  (direct tangent-space force)"""
        assert str(self.eq.G["f"]) == "I"

    def test_no_xi_dot_in_eom(self):
        assert not self.eqn.has(TimeDerivative(self.xi))

    def test_xi_not_in_eom_body(self):
        assert not self.eqn.has(self.xi)


# ---------------------------------------------------------------------------
# S2 formulation equivalence — ω·ω vs v·v
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# S2 spherical pendulum — qdot formulation (Lee et al. §5.3.1)
#
# KE = ½m‖v‖² with v = d/dt(l*q) = l*(ω×q)
#
# Expected EOM (Lee et al. Eq 5.7):
#   (I - qq^T){ml²ω̇ + mgl(q×e3)} = f
# which is equivalent to the ω formulation on the tangent space.
#
# Pipeline output:
#   Cross(q, Cross(d/dt(ω), q))*(-1)*l*l*m + Cross(q,e3)*m*g*l*(-1) + f
#
# The accel term is q×(ω̇×q) = (I-qq^T)ω̇ (projection), which equals
# ω̇ when ω̇ ⊥ q. This doesn't simplify further without the constraint
# system (issue #3).
#
# Standard form:
#   M[ω̇] = Hat(q)*Hat(q)*l*l*m = ml²*(qq^T - I) = -ml²*(I - qq^T)
#   f = Cross(q,e3)*m*g*l*(-1)  (same as ω formulation)
#   G[f] = I
# ---------------------------------------------------------------------------


class TestSphericalPendulumQdotPipeline:
    def setup_method(self):
        self.eom, self.variables, self.inputs, self.q = _spherical_pendulum_qdot()
        self.omega = self.q.get_tangent_vector()
        self.xi = self.q.get_variation_vector()
        self.key = list(self.eom.keys())[0]
        _, self.eqn = self.eom[self.key]
        self.sf = to_standard_form(self.eom, self.variables, self.inputs)
        self.eq = self.sf[self.key]

    def test_eom_str(self):
        """EOM: Cross(q, Cross(ω̇, q))*(-1)*l*l*m + Cross(q,e3)*mgl*(-1) + f"""
        expected = "Cross(q,Cross(\\frac{d}{dt}(\\omega_{q}),q))(-1)llm+Cross(q,e3)mgl(-1)+f"
        assert str(self.eqn) == f"({expected})"

    def test_eom_has_three_terms(self):
        assert len(self.eqn.nodes) == 3

    def test_M(self):
        """M[ω̇] = Hat(q)*Hat(q)*l*l*m  (= ml²(qq^T - I), the S2 projection)."""
        ddw = str(TimeDerivative(self.omega))
        assert ddw in self.eq.M
        assert str(self.eq.M[ddw]) == "Hat(q)Hat(q)llm"

    def test_f_same_as_omega_formulation(self):
        """f should be identical to the ω formulation: Cross(q,e3)*mgl*(-1)."""
        assert str(self.eq.f) == "Cross(q,e3)mgl(-1)"

    def test_G(self):
        assert str(self.eq.G["f"]) == "I"

    def test_no_xi_dot_in_eom(self):
        assert not self.eqn.has(TimeDerivative(self.xi))

    def test_no_spurious_coriolis(self):
        """No ω-dependent nonlinear terms (centripetal cancels for single S2 particle)."""
        omega_dot = TimeDerivative(self.omega)
        for n in self.eqn.nodes:
            if n.has(self.omega) and not n.has(omega_dot) and "e3" not in str(n) and str(n) != "f":
                assert False, f"Unexpected Coriolis-like term: {n}"


# ---------------------------------------------------------------------------
# SO3 free rigid body — Lee et al. Eq 6.16: Jω̇ + ω×Jω = M
#
# Pipeline output: -J*ω̇ - Cross(Ω, J*Ω) + M = 0
# (J' = J since J is symmetric, so J*(-0.5) + J'*(-0.5) = J*(-1))
# ---------------------------------------------------------------------------


class TestFreeRigidBodyPipeline:
    def setup_method(self):
        self.eom, self.variables, self.inputs, self.R = _rigid_body()
        self.Om = self.R.get_tangent_vector()
        self.eta = self.R.get_variation_vector()
        self.key = list(self.eom.keys())[0]
        _, self.eqn = self.eom[self.key]
        self.sf = to_standard_form(self.eom, self.variables, self.inputs)
        self.eq = self.sf[self.key]

    def test_eom_key_is_eta(self):
        assert str(self.eta) in self.eom

    def test_eom_str(self):
        """EOM: -J*ω̇ - ω×Jω + M = 0"""
        expected = "J\\frac{d}{dt}(\\Omega_{R})(-1.0)+Cross(\\Omega_{R},J\\Omega_{R})(-1)+M"
        assert str(self.eqn) == f"({expected})"

    def test_M(self):
        """M[Ω̇] = J*(-1)  (= -J, the inertia matrix)."""
        ddOm = str(TimeDerivative(self.Om))
        assert ddOm in self.eq.M
        assert str(self.eq.M[ddOm]) == "J(-1.0)"

    def test_f(self):
        """f = Cross(Ω, JΩ)*(-1)  (= -ω×Jω, the Coriolis term)."""
        assert str(self.eq.f) == "Cross(\\Omega_{R},J\\Omega_{R})(-1)"

    def test_G(self):
        """G[M] = I  (direct torque input)."""
        assert str(self.eq.G["M"]) == "I"

    def test_no_eta_dot_in_eom(self):
        assert not self.eqn.has(TimeDerivative(self.eta))

    def test_eta_not_in_eom_body(self):
        assert not self.eqn.has(self.eta)


# ---------------------------------------------------------------------------
# SO3 rigid pendulum — J about pivot (Lee et al. Eq 6.8)
#
# KE = ½ Ω^T J Ω,  PE = mg(Rρ)·e3
# Expected: Jω̇ + ω×Jω + mg*ρ×(R^T e3) = M
# Pipeline (all on LHS): -Jω̇ - ω×Jω - mg*ρ×(R^T e3) + M = 0
# ---------------------------------------------------------------------------


class TestRigidPendulumPivotPipeline:
    def setup_method(self):
        self.eom, self.variables, self.inputs, self.R = _rigid_pendulum_pivot()
        self.Om = self.R.get_tangent_vector()
        self.eta = self.R.get_variation_vector()
        self.key = list(self.eom.keys())[0]
        _, self.eqn = self.eom[self.key]
        self.sf = to_standard_form(self.eom, self.variables, self.inputs)
        self.eq = self.sf[self.key]

    def test_eom_str(self):
        """EOM: -Jω̇ - ω×Jω - mg*ρ×(R^T e3) + M = 0"""
        expected = (
            "J\\frac{d}{dt}(\\Omega_{R})(-1.0)"
            "+Cross(\\Omega_{R},J\\Omega_{R})(-1)"
            "+Cross(\\rho,(R)'e3)mg(-1)"
            "+M"
        )
        assert str(self.eqn) == f"({expected})"

    def test_M(self):
        """M[Ω̇] = -J  (same inertia as free rigid body)."""
        ddOm = str(TimeDerivative(self.Om))
        assert ddOm in self.eq.M
        assert str(self.eq.M[ddOm]) == "J(-1.0)"

    def test_f(self):
        """f = -ω×Jω - mg*ρ×(R^T e3).  Two terms: Coriolis + gravity."""
        f = self.eq.f
        assert hasattr(f, "nodes")
        assert len(f.nodes) == 2, f"Expected 2 terms, got {len(f.nodes)}: {f}"
        f_str = str(f)
        assert "Cross(\\Omega_{R},J\\Omega_{R})(-1)" in f_str
        assert "Cross(\\rho,(R)'e3)mg(-1)" in f_str

    def test_G(self):
        assert str(self.eq.G["M"]) == "I"

    def test_no_eta_dot(self):
        assert not self.eqn.has(TimeDerivative(self.eta))


# ---------------------------------------------------------------------------
# SO3 rigid pendulum — J about COM (parallel axis theorem)
#
# KE = ½ Ω^T J Ω + ½ m ‖v‖²,  v = d/dt(R*ρ)
# Effective inertia: J_eff = J + m*hat(ρ)²
# Expected: (J + m*hat(ρ)²)ω̇ + ω×Jω + m*ω×(ρ×(ω×ρ)) + mg*ρ×(R^T e3) = M
# ---------------------------------------------------------------------------


class TestRigidPendulumCOMPipeline:
    def setup_method(self):
        self.eom, self.variables, self.inputs, self.R = _rigid_pendulum_com()
        self.Om = self.R.get_tangent_vector()
        self.eta = self.R.get_variation_vector()
        self.key = list(self.eom.keys())[0]
        _, self.eqn = self.eom[self.key]
        self.sf = to_standard_form(self.eom, self.variables, self.inputs)
        self.eq = self.sf[self.key]

    def test_eom_str(self):
        """EOM: -Jω̇ - ω×Jω - m*ρ×(ω̇×ρ) - m*ω×(ρ×(ω×ρ)) - mg*ρ×(R^T e3) + M = 0

        The Jω̇ and m*ρ×(ω̇×ρ) terms combine in standard form as
        M = -(J + m*hat(ρ)²), since ρ×(ω̇×ρ) = -hat(ρ)²*ω̇.
        """
        expected = (
            "J\\frac{d}{dt}(\\Omega_{R})(-1.0)"
            "+Cross(\\Omega_{R},J\\Omega_{R})(-1)"
            "+Cross(\\rho,Cross(\\frac{d}{dt}(\\Omega_{R}),\\rho))(-1)m"
            "+Cross(\\Omega_{R},Cross(\\rho,Cross(\\Omega_{R},\\rho)))(-1)m"
            "+Cross(\\rho,(R)'e3)mg(-1)"
            "+M"
        )
        assert str(self.eqn) == f"({expected})"

    def test_M(self):
        """M = -(J + m*hat(ρ)²)  (COM inertia + parallel axis)."""
        ddOm = str(TimeDerivative(self.Om))
        assert ddOm in self.eq.M
        assert str(self.eq.M[ddOm]) == "(J(-1.0)+Hat(\\rho)Hat(\\rho)m)"

    def test_f_has_three_terms(self):
        """f has Coriolis + centripetal + gravity (3 terms)."""
        f = self.eq.f
        assert hasattr(f, "nodes")
        assert len(f.nodes) == 3, f"Expected 3 terms, got {len(f.nodes)}: {f}"

    def test_f_no_R_transpose_R(self):
        """R^T*R should have simplified to I."""
        assert "(R)'R" not in str(self.eq.f)

    def test_f_no_acceleration(self):
        assert str(TimeDerivative(self.Om)) not in str(self.eq.f)

    def test_G(self):
        assert str(self.eq.G["M"]) == "I"


# ---------------------------------------------------------------------------
# SO3 × SO3 double rigid pendulum
# ---------------------------------------------------------------------------


class TestDoubleRigidPendulumPipeline:
    def setup_method(self):
        self.eom, self.variables, self.inputs, self.R1, self.R2 = _double_rigid_pendulum()
        self.Om1 = self.R1.get_tangent_vector()
        self.Om2 = self.R2.get_tangent_vector()
        self.eta1 = self.R1.get_variation_vector()
        self.eta2 = self.R2.get_variation_vector()
        self.sf = to_standard_form(self.eom, self.variables, self.inputs)

    def test_produces_two_equations(self):
        assert len(self.eom) == 2

    def test_keys_are_eta1_eta2(self):
        assert str(self.eta1) in self.eom
        assert str(self.eta2) in self.eom

    def test_no_variation_dots(self):
        """After IBP, no d/dt(η₁) or d/dt(η₂) should remain."""
        for key, (_, eqn) in self.eom.items():
            assert not eqn.has(TimeDerivative(self.eta1)), f"d/dt(η₁) in {key}"
            assert not eqn.has(TimeDerivative(self.eta2)), f"d/dt(η₂) in {key}"

    def test_eta1_equation_M(self):
        """η₁ equation M should contain J1, ρ₁, l₁, m1, m2."""
        eq = self.sf[str(self.eta1)]
        ddOm1 = str(TimeDerivative(self.Om1))
        assert ddOm1 in eq.M
        M_str = str(eq.M[ddOm1])
        assert "J1" in M_str, f"M should contain J1: {M_str}"
        assert "\\rho_1" in M_str, f"M should contain ρ₁: {M_str}"
        assert "l_1" in M_str, f"M should contain l₁: {M_str}"
        assert "m1" in M_str, f"M should contain m1: {M_str}"
        assert "m2" in M_str, f"M should contain m2: {M_str}"

    def test_eta2_equation_M(self):
        """η₂ equation M should contain J2, ρ₂, m2."""
        eq = self.sf[str(self.eta2)]
        ddOm2 = str(TimeDerivative(self.Om2))
        assert ddOm2 in eq.M
        M_str = str(eq.M[ddOm2])
        assert "J2" in M_str, f"M should contain J2: {M_str}"
        assert "\\rho_2" in M_str, f"M should contain ρ₂: {M_str}"
        assert "m2" in M_str, f"M should contain m2: {M_str}"

    def test_eta1_f_has_gravity(self):
        eq = self.sf[str(self.eta1)]
        f_str = str(eq.f)
        assert "e3" in f_str, f"f should contain gravity: {f_str}"
        assert "g" in f_str, f"f should contain g: {f_str}"

    def test_eta2_f_has_gravity(self):
        eq = self.sf[str(self.eta2)]
        f_str = str(eq.f)
        assert "e3" in f_str, f"f should contain gravity: {f_str}"
        assert "g" in f_str, f"f should contain g: {f_str}"

    def test_G_maps_torques_independently(self):
        """Each equation's G should map only its own torque."""
        eq1 = self.sf[str(self.eta1)]
        eq2 = self.sf[str(self.eta2)]
        assert "M1" in eq1.G and str(eq1.G["M1"]) == "I"
        assert "M2" in eq2.G and str(eq2.G["M2"]) == "I"

    def test_standard_form_has_two_equations(self):
        assert len(self.sf) == 2


# ---------------------------------------------------------------------------
# R3 × SO3 mixed system (uncoupled point mass + rigid body)
#
# Translational: mẍ = F  (Newton)
# Rotational:    Jω̇ + ω×Jω = M  (Euler)
# ---------------------------------------------------------------------------


class TestMixedR3SO3Pipeline:
    def setup_method(self):
        self.eom, self.variables, self.inputs, self.x, self.R = _simple_se3_system()
        self.Om = self.R.get_tangent_vector()
        self.eta = self.R.get_variation_vector()
        self.sf = to_standard_form(self.eom, self.variables, self.inputs)

    def test_produces_two_equations(self):
        assert len(self.eom) == 2

    def test_translational_eom(self):
        """Translational: mẍ = F → M[ẍ] = -m*I, f = 0, G[F] = I."""
        key = str(Variation(self.x))
        assert key in self.sf
        eq = self.sf[key]
        ddx = str(TimeDerivative(TimeDerivative(self.x)))
        assert ddx in eq.M
        assert str(eq.M[ddx]) == "I(-1)m"
        assert str(eq.f) == "0v"
        assert str(eq.G["F"]) == "I"

    def test_rotational_eom(self):
        """Rotational: Jω̇ + ω×Jω = M → same as free rigid body."""
        key = str(self.eta)
        assert key in self.sf
        eq = self.sf[key]
        ddOm = str(TimeDerivative(self.Om))
        assert ddOm in eq.M
        M_str = str(eq.M[ddOm])
        assert "J" in M_str
        assert str(eq.G["M"]) == "I"
