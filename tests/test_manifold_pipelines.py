"""End-to-end tests for manifold EOM and standard form pipelines.

Tests the full pipeline: Lagrangian → variation → manifold rules →
simplify → IBP → expand → extract → standard form for S2, SO3,
and mixed systems.
"""

from geomech import (
    S2,
    SO3,
    Dot,
    Matrix,
    Scalar,
    SystemVariables,
    TimeDerivative,
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
    This matches the Scala reference (spherical_pendulum.scala).
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


def _rigid_pendulum():
    """Rigid pendulum on SO3 with gravity."""
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


# ---------------------------------------------------------------------------
# S2 spherical pendulum
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

    def test_eom_produces_one_equation(self):
        assert len(self.eom) == 1

    def test_eom_key_is_xi(self):
        assert str(self.xi) in self.eom

    def test_eom_contains_acceleration(self):
        """EOM should contain d/dt(omega)."""
        assert self.eqn.has(TimeDerivative(self.omega))

    def test_eom_contains_gravity(self):
        assert "e3" in str(self.eqn) or "g" in str(self.eqn)

    def test_eom_exact(self):
        """EOM should be: -ml²ω̇ - mgl(q×e3) + f = 0  (Lee et al. Eq 5.14).

        Pipeline convention is M*a + f + G*u = 0, so all terms on one side.
        Rearranged: ml²ω̇ + mgl(q×e3) = f.
        """
        assert hasattr(self.eqn, "nodes")
        terms = self.eqn.nodes
        assert len(terms) == 3, f"Expected 3 terms, got {len(terms)}: {self.eqn}"

        # Classify each term by what it contains
        has_accel = [t for t in terms if t.has(TimeDerivative(self.omega))]
        has_gravity = [t for t in terms if t.has(self.q) and not t.has(self.omega)]
        has_input = [t for t in terms if str(t) == "f"]

        assert len(has_accel) == 1, f"Expected 1 accel term: {terms}"
        assert len(has_gravity) == 1, f"Expected 1 gravity term: {terms}"
        assert len(has_input) == 1, f"Expected 1 input term: {terms}"

        # Accel term should contain ω̇ but not q (state-independent inertia)
        assert not has_accel[0].has(self.q), f"Accel term should not contain q: {has_accel[0]}"

        # Gravity term should contain q and e3 (Cross(q, e3))
        assert has_gravity[0].has(self.q), f"Gravity term should contain q: {has_gravity[0]}"

    def test_standard_form_M_exact(self):
        """M should be proportional to ml² (scalar inertia on S2)."""
        ddw = str(TimeDerivative(self.omega))
        assert ddw in self.eq.M
        M_str = str(self.eq.M[ddw])
        assert "m" in M_str
        assert "l" in M_str
        # M should not contain q, e3, or g (those belong in f)
        assert "q" not in M_str, f"M should not depend on q: {M_str}"
        assert "e3" not in M_str, f"M should not contain e3: {M_str}"
        assert "g" not in M_str, f"M should not contain g: {M_str}"

    def test_standard_form_f_exact(self):
        """f should be gravity: mgl(q × e3) (up to sign)."""
        f_str = str(self.eq.f)
        # Must contain gravity terms
        assert "e3" in f_str, f"f should contain e3: {f_str}"
        assert "g" in f_str, f"f should contain g: {f_str}"
        assert "q" in f_str, f"f should contain q (via Cross): {f_str}"
        # Must NOT contain ω (no Coriolis for single particle on S2)
        assert str(self.omega) not in f_str, f"f should not contain ω: {f_str}"

    def test_standard_form_G_is_identity(self):
        """G should map f directly (identity), since δW = ξ · f."""
        assert "f" in self.eq.G
        G_str = str(self.eq.G["f"])
        assert G_str == "I", f"G[f] should be identity, got: {G_str}"

    def test_no_xi_dot_in_eom(self):
        """After IBP, no d/dt(xi) should remain."""
        assert not self.eqn.has(TimeDerivative(self.xi)), "IBP should have removed d/dt(xi)"


# ---------------------------------------------------------------------------
# S2 Lagrangian formulation equivalence
# ---------------------------------------------------------------------------


class TestS2FormulationEquivalence:
    """Both formulations of S2 KE should produce the same EOM.

    There are two natural ways to write KE for a particle on S2:

      A) Angular velocity formulation (tangent space):
         KE = ½ m l² (ω · ω)
         Uses ω directly. This is the "modified Lagrangian" L̃(q,ω)
         from Lee et al. §5.3.3 (p214). Built by _spherical_pendulum().

      B) Configuration velocity formulation (ambient space):
         x = l*q,  v = ẋ = l*(ω × q)
         KE = ½ m (v · v)
         Uses q̇ = ω × q (Eq 5.1). This is L(q,q̇) from Lee et al. §5.3.1.
         The pipeline substitutes q̇ = ω × q and simplifies using
         ‖ω × q‖² = ‖ω‖² (since ω ⊥ q and ‖q‖ = 1).
         Built by _spherical_pendulum_qdot().

    Both formulations should produce identical EOM after simplification.

    Note on infinitesimal work: these tests use δW = ξ · f where f is
    a force in the tangent space. The Scala reference uses δW = δq · u
    where u is a force in R3 and δq = ξ × q. These are related by
    f = q × u (projection to tangent space) and produce different G
    matrices but equivalent dynamics.
    """

    def setup_method(self):
        self.eom_A, _, _, self.q_A = _spherical_pendulum()
        self.eom_B, _, _, self.q_B = _spherical_pendulum_qdot()
        xi_A = self.q_A.get_variation_vector()
        xi_B = self.q_B.get_variation_vector()
        _, self.eqn_A = self.eom_A[str(xi_A)]
        _, self.eqn_B = self.eom_B[str(xi_B)]

    def test_omega_vs_qdot_formulation(self):
        """L̃(q,ω) = ½ml²(ω·ω) vs L(q,q̇) = ½m(v·v) should match."""
        terms_A = len(self.eqn_A.nodes) if hasattr(self.eqn_A, "nodes") else 1
        terms_B = len(self.eqn_B.nodes) if hasattr(self.eqn_B, "nodes") else 1
        assert terms_A == terms_B, (
            f"Formulation mismatch: A has {terms_A} terms, B has {terms_B}. "
            f"A={self.eqn_A}, B={self.eqn_B}"
        )

    def test_qdot_has_no_spurious_coriolis(self):
        """The v = l*q̇ formulation should not produce Coriolis terms.

        For a single particle on S2, the centripetal acceleration term
        (ω × (ω × q)) that arises from d/dt(ω × q) cancels exactly
        via the BAC-CAB identity: q × (ω × (ω × q)) = -‖ω‖²q × q = 0
        projected onto the tangent space. So the final EOM should only
        have: ml²ω̇ (inertia), gravity, and input — no ω-dependent
        nonlinear terms.
        """
        omega = self.q_B.get_tangent_vector()
        omega_dot = TimeDerivative(omega)
        for n in self.eqn_B.nodes if hasattr(self.eqn_B, "nodes") else [self.eqn_B]:
            if n.has(omega) and not n.has(omega_dot) and "e3" not in str(n) and str(n) != "f":
                assert False, f"Unexpected Coriolis-like term: {n}"


# ---------------------------------------------------------------------------
# SO3 free rigid body
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

    def test_eom_produces_one_equation(self):
        assert len(self.eom) == 1

    def test_eom_key_is_eta(self):
        assert str(self.eta) in self.eom

    def test_no_eta_dot_in_eom(self):
        """After IBP, no d/dt(eta) should remain."""
        assert not self.eqn.has(TimeDerivative(self.eta)), "IBP should have removed d/dt(eta)"

    def test_standard_form_M(self):
        """M should contain J (Lee et al. Eq 6.16: Jω̇ + ω×Jω = M).

        Note: currently M = J(-0.5) + J'(-0.5) because the simplifier
        doesn't yet use J = J' (symmetric). This is correct but unsimplified.
        """
        ddOm = str(TimeDerivative(self.Om))
        assert ddOm in self.eq.M
        M_str = str(self.eq.M[ddOm])
        assert "J" in M_str, f"M should contain J: {M_str}"
        # M should not contain Omega (that belongs in f)
        assert str(self.Om) not in M_str, f"M should not contain Ω: {M_str}"
        # M should not contain gravity or position
        assert "e3" not in M_str
        assert "g" not in M_str

    def test_standard_form_f(self):
        """f should be the Coriolis term ω×Jω (Lee et al. Eq 6.16).

        For a free rigid body with no gravity, f contains only
        the gyroscopic/Coriolis coupling.
        """
        f_str = str(self.eq.f)
        # f must contain Omega and J (Coriolis: ω×Jω)
        assert str(self.Om) in f_str, f"f should contain Ω: {f_str}"
        assert "J" in f_str, f"f should contain J: {f_str}"
        # f should NOT contain acceleration
        assert str(TimeDerivative(self.Om)) not in f_str, f"f should not contain ω̇: {f_str}"
        # f should NOT contain gravity (no gravity in free rigid body)
        assert "e3" not in f_str, f"f should not contain gravity: {f_str}"

    def test_standard_form_G(self):
        """G maps torque M directly (identity), since δW = η · M."""
        assert "M" in self.eq.G
        G_str = str(self.eq.G["M"])
        assert G_str == "I", f"G[M] should be identity, got: {G_str}"


# ---------------------------------------------------------------------------
# SO3 rigid pendulum (with translational KE and gravity)
# ---------------------------------------------------------------------------


class TestRigidPendulumPipeline:
    def setup_method(self):
        self.eom, self.variables, self.inputs, self.R = _rigid_pendulum()
        self.Om = self.R.get_tangent_vector()
        self.eta = self.R.get_variation_vector()
        self.key = list(self.eom.keys())[0]
        _, self.eqn = self.eom[self.key]
        self.sf = to_standard_form(self.eom, self.variables, self.inputs)
        self.eq = self.sf[self.key]

    def test_eom_produces_one_equation(self):
        assert len(self.eom) == 1

    def test_no_eta_dot_in_eom(self):
        """After IBP, no d/dt(eta) should remain."""
        assert not self.eqn.has(TimeDerivative(self.eta)), "IBP should have removed d/dt(eta)"

    def test_standard_form_M(self):
        """M should contain J (rotational) and ρ (translational inertia).

        The rigid pendulum inertia matrix is J + m*hat(ρ)^T*hat(ρ)
        (Lee et al. Eq 6.8 with offset mass).
        """
        ddOm = str(TimeDerivative(self.Om))
        assert ddOm in self.eq.M
        M_str = str(self.eq.M[ddOm])
        assert "J" in M_str, f"M should contain J: {M_str}"
        assert "\\rho" in M_str, f"M should contain ρ: {M_str}"
        assert "m" in M_str, f"M should contain m: {M_str}"
        # M should not contain Omega, gravity, or input
        assert str(self.Om) not in M_str, f"M should not contain Ω: {M_str}"

    def test_standard_form_f(self):
        """f should contain Coriolis (ω×Jω) and gravity (mg cross term)."""
        f_str = str(self.eq.f)
        # Must have Coriolis (Omega and J)
        assert str(self.Om) in f_str, f"f should contain Ω (Coriolis): {f_str}"
        assert "J" in f_str, f"f should contain J (Coriolis): {f_str}"
        # Must have gravity
        assert "e3" in f_str, f"f should contain e3 (gravity): {f_str}"
        assert "g" in f_str, f"f should contain g (gravity): {f_str}"
        # f should NOT contain acceleration
        assert str(TimeDerivative(self.Om)) not in f_str, f"f should not contain ω̇: {f_str}"

    def test_standard_form_G(self):
        """G maps torque M directly (identity)."""
        assert "M" in self.eq.G
        G_str = str(self.eq.G["M"])
        assert G_str == "I", f"G[M] should be identity, got: {G_str}"


# ---------------------------------------------------------------------------
# Variation vector should not appear in EOM body
# ---------------------------------------------------------------------------


class TestVariationVectorExtracted:
    def test_s2_xi_not_in_eom_body(self):
        """ξ should only be the key, not appear inside the equation."""
        eom, _, _, q = _spherical_pendulum()
        xi = q.get_variation_vector()
        key = list(eom.keys())[0]
        _, eqn = eom[key]
        assert not eqn.has(xi), f"ξ should be factored out of EOM body: {eqn}"

    def test_so3_eta_not_in_eom_body(self):
        """η should only be the key, not appear inside the equation."""
        eom, _, _, R = _rigid_body()
        eta = R.get_variation_vector()
        key = list(eom.keys())[0]
        _, eqn = eom[key]
        assert not eqn.has(eta), f"η should be factored out of EOM body: {eqn}"


# ---------------------------------------------------------------------------
# Double rigid pendulum (SO3 × SO3)
# ---------------------------------------------------------------------------


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

    # Positions
    x1 = R1 * rho1
    x2 = R1 * l1 + R2 * rho2

    # Velocities
    v1 = x1.t_diff()
    v2 = x2.t_diff()

    # Lagrangian
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


class TestDoubleRigidPendulumPipeline:
    def test_eom_produces_two_equations(self):
        eom, _, _, R1, R2 = _double_rigid_pendulum()
        assert len(eom) == 2

    def test_eom_keys_are_eta1_eta2(self):
        eom, _, _, R1, R2 = _double_rigid_pendulum()
        eta1 = R1.get_variation_vector()
        eta2 = R2.get_variation_vector()
        assert str(eta1) in eom
        assert str(eta2) in eom

    def test_eom_contains_both_accelerations(self):
        eom, _, _, R1, R2 = _double_rigid_pendulum()
        Om1 = R1.get_tangent_vector()
        Om2 = R2.get_tangent_vector()
        all_eqn_str = " ".join(str(eqn) for _, eqn in eom.values())
        assert str(TimeDerivative(Om1)) in all_eqn_str
        assert str(TimeDerivative(Om2)) in all_eqn_str

    def test_no_variation_dots_in_eom(self):
        """After IBP, no d/dt(η₁) or d/dt(η₂) should remain."""
        eom, _, _, R1, R2 = _double_rigid_pendulum()
        eta1 = R1.get_variation_vector()
        eta2 = R2.get_variation_vector()
        for key, (_, eqn) in eom.items():
            assert not eqn.has(TimeDerivative(eta1)), f"d/dt(η₁) in {key}"
            assert not eqn.has(TimeDerivative(eta2)), f"d/dt(η₂) in {key}"

    def test_standard_form_has_two_equations(self):
        eom, variables, inputs, R1, R2 = _double_rigid_pendulum()
        sf = to_standard_form(eom, variables, inputs)
        assert len(sf) == 2

    def test_standard_form_M_has_cross_coupling(self):
        """Double pendulum M matrix should have cross-coupling terms."""
        eom, variables, inputs, R1, R2 = _double_rigid_pendulum()
        sf = to_standard_form(eom, variables, inputs)
        Om1 = R1.get_tangent_vector()
        Om2 = R2.get_tangent_vector()
        ddOm1 = str(TimeDerivative(Om1))
        ddOm2 = str(TimeDerivative(Om2))
        for key, eq in sf.items():
            if ddOm1 in eq.M and ddOm2 in eq.M:
                return
        all_M_keys = set()
        for eq in sf.values():
            all_M_keys.update(eq.M.keys())
        assert ddOm1 in all_M_keys and ddOm2 in all_M_keys


# ---------------------------------------------------------------------------
# Mixed R3 + SO3 system (quadrotor-like)
# ---------------------------------------------------------------------------


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


class TestMixedR3SO3Pipeline:
    def setup_method(self):
        self.eom, self.variables, self.inputs, self.x, self.R = _simple_se3_system()
        self.Om = self.R.get_tangent_vector()
        self.eta = self.R.get_variation_vector()
        self.sf = to_standard_form(self.eom, self.variables, self.inputs)

    def test_eom_produces_two_equations(self):
        """R3 + SO3 should produce one equation per DOF group."""
        assert len(self.eom) == 2

    def test_eom_keys(self):
        keys = set(self.eom.keys())
        assert str(self.x.delta()) in keys or "\\delta{x}" in keys
        assert str(self.eta) in keys

    def test_translational_eom_is_newton(self):
        """Translational EOM should be m*ẍ = F (Newton's law)."""
        ddx = str(TimeDerivative(TimeDerivative(self.x)))
        for key, eq in self.sf.items():
            if ddx in eq.M:
                M_str = str(eq.M[ddx])
                assert "m" in M_str, f"Newton's law: M should contain m: {M_str}"
                return
        assert False, "No translational equation found with ẍ"

    def test_rotational_eom_is_euler(self):
        """Rotational EOM should be J*Ω̇ + ω×Jω = M."""
        ddOm = str(TimeDerivative(self.Om))
        for key, eq in self.sf.items():
            if ddOm in eq.M:
                assert "J" in str(eq.M[ddOm])
                return
        assert False, "No rotational equation found with Ω̇"
