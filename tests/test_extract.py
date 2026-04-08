"""Tests for geomech.core.math.extract — coefficient extraction."""

import pytest

from geomech.core.base.expressions import (
    Matrix,
    Scalar,
    Vector,
    ZeroMatrix,
)
from geomech.core.math.extract import (
    _vector_from_matrix,
    _vector_from_scalar,
    _vector_from_vector,
    extract_linear_coeff,
)
from geomech.core.operations.addition import Add, MAdd, VAdd
from geomech.core.operations.geometry import Cross, Dot, Hat, Transpose
from geomech.core.operations.multiplication import (
    MMMul,
    Mul,
    MVMul,
    SMMul,
    SVMul,
    VVMul,
)
from geomech.utils.errors import AlgebraicError, NonLinearError

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def scalars():
    a = Scalar("a")
    b = Scalar("b")
    c = Scalar("c", value=3, attr=["Constant"])
    return a, b, c


@pytest.fixture
def vectors():
    v = Vector("v")
    w = Vector("w")
    u = Vector("u")
    return v, w, u


@pytest.fixture
def matrices():
    M = Matrix("M")
    N = Matrix("N")
    return M, N


# ===========================================================================
# extract_linear_coeff dispatch
# ===========================================================================


class TestExtractCoeffDispatch:
    """extract_linear_coeff routes to the right extractor based on expr.type."""

    def test_scalar_dispatches_to__vector_from_scalar(self, vectors):
        """Scalar expression dispatches to _vector_from_scalar."""
        v, w, _ = vectors
        expr = Dot(v, w)
        result = extract_linear_coeff(expr, v)
        assert str(result) == str(w)

    def test_vector_dispatches_to__vector_from_vector(self, vectors, matrices):
        """Vector expression dispatches to _vector_from_vector."""
        v, _, _ = vectors
        M, _ = matrices
        expr = MVMul(M, v)
        result = extract_linear_coeff(expr, v)
        assert str(result) == str(M)

    def test_matrix_dispatches_to__vector_from_matrix(self, matrices):
        """Matrix expression dispatches to _vector_from_matrix."""
        M, N = matrices
        expr = MMMul(M, N)
        result = extract_linear_coeff(expr, N)
        assert str(result) == str(M)


# ===========================================================================
# _vector_from_scalar — Extract From Scalar
# ===========================================================================


class TestExtractFromScalarAdd:
    """_vector_from_scalar through scalar addition."""

    def test_single_dot_in_add(self, vectors):
        """Add with one term containing vec extracts that term only."""
        v, w, u = vectors
        a = Scalar("a")
        # Dot(v, w) + a  →  _vector_from_scalar extracts from Dot(v, w) only
        expr = Add(Dot(v, w), a)
        result = _vector_from_scalar(expr, v)
        assert str(result) == str(w)

    def test_multiple_dots_in_add(self, vectors):
        """Add with multiple terms containing vec combines via VAdd."""
        v, w, u = vectors
        expr = Add(Dot(v, w), Dot(v, u))
        result = _vector_from_scalar(expr, v)
        assert isinstance(result, VAdd)
        assert len(result.nodes) == 2

    def test_no_terms_with_vec(self, vectors, scalars):
        """Add where no term contains vec returns ZeroVector."""
        v, w, u = vectors
        a, b, _ = scalars
        expr = Add(a, b)
        result = _vector_from_scalar(expr, v)
        assert result.is_zero


class TestExtractFromScalarMul:
    """_vector_from_scalar through scalar multiplication."""

    def test_vec_in_left(self, vectors):
        """Mul(dot(v,w), s) → SVMul(_vector_from_scalar(dot(v,w), v), s) = SVMul(w, s)."""
        v, w, _ = vectors
        s = Scalar("s")
        expr = Mul(Dot(v, w), s)
        result = _vector_from_scalar(expr, v)
        assert isinstance(result, SVMul)
        assert str(result.right) == "s"

    def test_vec_in_right(self, vectors):
        """Mul(s, dot(v,w)) → SVMul(_vector_from_scalar(dot(v,w), v), s) = SVMul(w, s)."""
        v, w, _ = vectors
        s = Scalar("s")
        expr = Mul(s, Dot(v, w))
        result = _vector_from_scalar(expr, v)
        assert isinstance(result, SVMul)
        assert str(result.right) == "s"

    def test_neither_side_has_vec(self, vectors, scalars):
        """Mul(a, b) where neither contains vec returns ZeroVector."""
        v, _, _ = vectors
        a, b, _ = scalars
        expr = Mul(a, b)
        result = _vector_from_scalar(expr, v)
        assert result.is_zero


class TestExtractFromScalarDot:
    """_vector_from_scalar through dot product — the core extraction case."""

    def test_left_is_vec(self, vectors):
        """Dot(vec, w) → coefficient is w."""
        v, w, _ = vectors
        expr = Dot(v, w)
        result = _vector_from_scalar(expr, v)
        assert str(result) == str(w)

    def test_right_is_vec(self, vectors):
        """Dot(w, vec) → coefficient is w."""
        v, w, _ = vectors
        expr = Dot(w, v)
        result = _vector_from_scalar(expr, v)
        assert str(result) == str(w)

    def test_left_is_cross_containing_vec(self, vectors):
        """Dot(Cross(a, vec), b) → MVMul(Transpose(_vector_from_vector(Cross(a,vec), vec)), b).

        _vector_from_vector(Cross(a, vec), vec) = Hat(a), so result = Transpose(Hat(a)) * b.
        """
        v, w, u = vectors
        expr = Dot(Cross(w, v), u)
        result = _vector_from_scalar(expr, v)
        # _vector_from_vector(Cross(w, v), v) = Hat(w), so result = MVMul(Transpose(Hat(w)), u)
        assert isinstance(result, MVMul)
        assert isinstance(result.left, Transpose)

    def test_right_is_cross_containing_vec(self, vectors):
        """Dot(b, Cross(a, vec)) → MVMul(Transpose(_vector_from_vector(Cross(a,vec), vec)), b)."""
        v, w, u = vectors
        expr = Dot(u, Cross(w, v))
        result = _vector_from_scalar(expr, v)
        assert isinstance(result, MVMul)
        assert isinstance(result.left, Transpose)

    def test_neither_side_has_vec(self, vectors):
        """Dot(a, b) where neither contains vec returns ZeroVector."""
        v, w, u = vectors
        expr = Dot(w, u)
        result = _vector_from_scalar(expr, v)
        assert result.is_zero

    def test_both_sides_have_vec_raises(self, vectors):
        """Dot(f(vec), g(vec)) raises NotImplementedError."""
        v, w, _ = vectors
        expr = Dot(Cross(w, v), Cross(v, w))
        with pytest.raises(NonLinearError, match="nonlinear"):
            _vector_from_scalar(expr, v)


class TestExtractFromScalarLeaf:
    """_vector_from_scalar on leaf / unhandled expressions."""

    def test_plain_scalar_returns_zero(self, vectors, scalars):
        """Plain scalar leaf returns ZeroVector."""
        v, _, _ = vectors
        a, _, _ = scalars
        result = _vector_from_scalar(a, v)
        assert result.is_zero


# ===========================================================================
# _vector_from_vector — Extract From Vector
# ===========================================================================


class TestExtractFromVectorVAdd:
    """_vector_from_vector through vector addition."""

    def test_single_term_with_vec(self, vectors, matrices):
        """VAdd with one term containing vec extracts that term."""
        v, w, _ = vectors
        M, _ = matrices
        expr = VAdd(MVMul(M, v), w)
        result = _vector_from_vector(expr, v)
        assert str(result) == str(M)

    def test_multiple_terms_with_vec(self, vectors, matrices):
        """VAdd with multiple terms containing vec combines via MAdd."""
        v, _, _ = vectors
        M, N = matrices
        expr = VAdd(MVMul(M, v), MVMul(N, v))
        result = _vector_from_vector(expr, v)
        assert isinstance(result, MAdd)
        assert len(result.nodes) == 2

    def test_no_terms_with_vec(self, vectors, matrices):
        """VAdd where no term contains vec returns ZeroMatrix."""
        v, w, u = vectors
        expr = VAdd(w, u)
        result = _vector_from_vector(expr, v)
        assert result.is_zero


class TestExtractFromVectorCross:
    """_vector_from_vector through cross product."""

    def test_cross_right_is_vec(self, vectors):
        """Cross(a, vec) = Hat(a) * vec → coefficient is Hat(a)."""
        v, w, _ = vectors
        result = _vector_from_vector(Cross(w, v), v)
        assert isinstance(result, Hat)
        assert str(result.expr) == "w"

    def test_cross_left_is_vec(self, vectors):
        """Cross(vec, b) = -Hat(b) * vec → coefficient is -Hat(b)."""
        v, w, _ = vectors
        result = _vector_from_vector(Cross(v, w), v)
        assert isinstance(result, SMMul)
        assert isinstance(result.left, Hat)
        assert str(result.left.expr) == "w"

    def test_cross_self(self, vectors):
        """Cross(v, v) = 0 → coefficient is ZeroMatrix."""
        v, _, _ = vectors
        result = _vector_from_vector(Cross(v, v), v)
        assert result.is_zero

    def test_cross_left_contains_vec_recursive(self, vectors, matrices):
        """Cross(M*vec, b) → -Hat(b) * _vector_from_vector(M*vec, vec) = -Hat(b) * M."""
        v, w, _ = vectors
        M, _ = matrices
        expr = Cross(MVMul(M, v), w)
        result = _vector_from_vector(expr, v)
        assert isinstance(result, MMMul)
        # left should be SMMul(Hat(w), -1), right should be M
        assert isinstance(result.left, SMMul)
        assert str(result.right) == str(M)

    def test_cross_right_contains_vec_recursive(self, vectors, matrices):
        """Cross(a, M*vec) → Hat(a) * _vector_from_vector(M*vec, vec) = Hat(a) * M."""
        v, w, _ = vectors
        M, _ = matrices
        expr = Cross(w, MVMul(M, v))
        result = _vector_from_vector(expr, v)
        assert isinstance(result, MMMul)
        assert isinstance(result.left, Hat)
        assert str(result.right) == str(M)

    def test_cross_neither_has_vec(self, vectors):
        """Cross(a, b) where neither is vec returns ZeroMatrix."""
        v, w, u = vectors
        result = _vector_from_vector(Cross(w, u), v)
        assert result.is_zero


class TestExtractFromVectorMVMul:
    """_vector_from_vector through matrix-vector multiplication."""

    def test_right_is_vec(self, vectors, matrices):
        """MVMul(M, vec) → coefficient is M."""
        v, _, _ = vectors
        M, _ = matrices
        result = _vector_from_vector(MVMul(M, v), v)
        assert str(result) == str(M)

    def test_right_contains_vec(self, vectors, matrices):
        """MVMul(M, Cross(a, vec)) → MMMul(M, Hat(a))."""
        v, w, _ = vectors
        M, _ = matrices
        expr = MVMul(M, Cross(w, v))
        result = _vector_from_vector(expr, v)
        assert isinstance(result, MMMul)
        assert str(result.left) == str(M)
        assert isinstance(result.right, Hat)

    def test_neither_side_has_vec(self, vectors, matrices):
        """MVMul(M, w) where w is not vec returns ZeroMatrix."""
        v, w, _ = vectors
        M, _ = matrices
        result = _vector_from_vector(MVMul(M, w), v)
        assert result.is_zero


class TestExtractFromVectorMVMulMatSide:
    """_vector_from_vector through MVMul where the matrix side contains vec."""

    def test_hat_of_vec(self, vectors):
        """MVMul(Hat(vec), b) = cross(vec, b) → -Hat(b)."""
        v, w, _ = vectors
        expr = MVMul(Hat(v), w)
        result = _vector_from_vector(expr, v)
        assert isinstance(result, SMMul)
        assert isinstance(result.left, Hat)
        assert str(result.left.expr) == "w"

    def test_hat_of_expr_containing_vec(self, vectors, matrices):
        """MVMul(Hat(M*vec), b) → -Hat(b) * _vector_from_vector(M*vec, vec) = -Hat(b) * M."""
        v, w, _ = vectors
        M, _ = matrices
        expr = MVMul(Hat(MVMul(M, v)), w)
        result = _vector_from_vector(expr, v)
        assert isinstance(result, MMMul)
        assert isinstance(result.left, SMMul)
        assert str(result.right) == str(M)

    def test_mmmul_right_has_vec(self, vectors, matrices):
        """MVMul(MMMul(A, B), b) where B has vec → A * _vector_from_vector(MVMul(B, b), vec)."""
        v, w, _ = vectors
        M, N = matrices
        # MVMul(MMMul(M, Hat(v)), w) — Hat(v) contains v
        # → MMMul(M, _vector_from_vector(MVMul(Hat(v), w), v))
        # → MMMul(M, SMMul(Hat(w), -1))
        expr = MVMul(MMMul(M, Hat(v)), w)
        result = _vector_from_vector(expr, v)
        assert isinstance(result, MMMul)
        assert str(result.left) == str(M)

    def test_smmul_mat_has_vec(self, vectors):
        """MVMul(SMMul(Hat(vec), s), b) redistributes scalar then extracts.

        SMMul(Hat(v), s) * w → MVMul(Hat(v), SVMul(w, s))
        → Hat case: expr == v → SMMul(Hat(SVMul(w,s)), -1)
        """
        v, w, _ = vectors
        s = Scalar("s")
        expr = MVMul(SMMul(Hat(v), s), w)
        result = _vector_from_vector(expr, v)
        assert isinstance(result, SMMul)
        assert isinstance(result.left, Hat)


class TestExtractFromVectorMVMulMAddMat:
    """_vector_from_vector through MVMul where the matrix side is an MAdd."""

    def test_madd_one_term_has_vec(self, vectors, matrices):
        """MVMul(MAdd(Hat(vec), N), b) → extracts from the Hat(vec) term only."""
        v, w, _ = vectors
        _, N = matrices
        expr = MVMul(MAdd(Hat(v), N), w)
        result = _vector_from_vector(expr, v)
        # Hat(v) has v, N doesn't → _vector_from_vector(MVMul(Hat(v), w), v) → SMMul(Hat(w), -1)
        assert isinstance(result, SMMul)
        assert isinstance(result.left, Hat)

    def test_madd_both_terms_have_vec(self, vectors):
        """MVMul(MAdd(Hat(v), SMMul(Hat(v),-1)), w) → MAdd of two coefficients."""
        v, w, _ = vectors
        mat = MAdd(Hat(v), SMMul(Hat(v), -1))
        expr = MVMul(mat, w)
        result = _vector_from_vector(expr, v)
        assert isinstance(result, MAdd)
        assert len(result.nodes) == 2


class TestExtractFromVectorMVMulTransposeMat:
    """_vector_from_vector through MVMul where the matrix side is a Transpose."""

    def test_transpose_hat(self, vectors):
        """MVMul(Tr(Hat(v)), w) → rewrite to MVMul(Hat(v), SVMul(w, -1)) and extract."""
        v, w, _ = vectors
        expr = MVMul(Transpose(Hat(v)), w)
        result = _vector_from_vector(expr, v)
        # Hat(v)^T = -Hat(v), so Tr(Hat(v))*w = -Hat(v)*w
        # → extract v from MVMul(Hat(v), SVMul(w, -1)) → SMMul(Hat(SVMul(w,-1)), -1)
        assert isinstance(result, SMMul)

    def test_transpose_mmmul(self, vectors, matrices):
        """MVMul(Tr(MMMul(A, Hat(v))), w) → MVMul(Tr(Hat(v)), MVMul(Tr(A), w)) and extract."""
        v, w, _ = vectors
        M, _ = matrices
        expr = MVMul(Transpose(MMMul(M, Hat(v))), w)
        result = _vector_from_vector(expr, v)
        # Tr(M * Hat(v)) * w = Hat(v)^T * M^T * w → extract v
        assert result is not ZeroMatrix


class TestExtractFromVectorLeaf:
    """_vector_from_vector on leaf expressions."""

    def test_plain_vector_not_target_is_zero(self, vectors):
        """Plain Vector that isn't the target has zero coefficient."""
        v, w, _ = vectors
        result = _vector_from_vector(w, v)
        assert result.is_zero


class TestExtractFromVectorSVMul:
    """_vector_from_vector through scalar-vector multiplication."""

    def test_vec_side_contains_vec(self, vectors, matrices):
        """SVMul(M*vec, s) → SMMul(_vector_from_vector(M*vec, vec), s) = SMMul(M, s)."""
        v, _, _ = vectors
        M, _ = matrices
        s = Scalar("s")
        expr = SVMul(MVMul(M, v), s)
        result = _vector_from_vector(expr, v)
        assert isinstance(result, SMMul)
        assert str(result.right) == "s"

    def test_scalar_side_dot_redistribution(self, vectors):
        """SVMul(w, Dot(v, u)) extract v → MVMul(VVMul(w, Tr(u)), v) → VVMul(w, Tr(u))."""
        v, w, u = vectors
        expr = SVMul(w, Dot(v, u))
        result = _vector_from_vector(expr, v)
        # Redistributed to MVMul(VVMul(w, Transpose(u)), v), coefficient = VVMul(w, Tr(u))
        assert isinstance(result, VVMul)
        assert str(result.left) == "w"
        assert isinstance(result.right, Transpose)

    def test_scalar_side_mul_redistribution(self, vectors):
        """SVMul(w, Mul(a, Dot(v, u))) extract v → redistributes through Mul."""
        v, w, u = vectors
        a = Scalar("a")
        expr = SVMul(w, Mul(a, Dot(v, u)))
        result = _vector_from_vector(expr, v)
        # SVMul(SVMul(w, a), Dot(v, u)) → MVMul(VVMul(SVMul(w,a), Tr(u)), v)
        assert isinstance(result, VVMul)

    def test_scalar_side_add_redistribution(self, vectors):
        """SVMul(w, Add(Dot(v,u), Dot(v,w))) extract v → MAdd of two coefficients."""
        v, w, u = vectors
        expr = SVMul(w, Add(Dot(v, u), Dot(v, w)))
        result = _vector_from_vector(expr, v)
        assert isinstance(result, MAdd)
        assert len(result.nodes) == 2


# ===========================================================================
# _vector_from_matrix — Extract From Matrix
# ===========================================================================


class TestExtractFromMatrix:
    """_vector_from_matrix — extract from matrix."""

    def test_madd_one_matching_term(self, matrices):
        """MAdd with one term containing target extracts it."""
        M, N = matrices
        P = Matrix("P")
        expr = MAdd(MMMul(M, N), P)
        result = _vector_from_matrix(expr, N)
        assert str(result) == str(M)

    def test_madd_multiple_matching_terms(self, matrices):
        """MAdd with multiple terms containing target combines via MAdd."""
        M, N = matrices
        P = Matrix("P")
        expr = MAdd(MMMul(M, N), MMMul(P, N))
        result = _vector_from_matrix(expr, N)
        assert isinstance(result, MAdd)
        assert len(result.nodes) == 2

    def test_madd_no_matching_terms(self, matrices):
        """MAdd where no term contains target returns ZeroMatrix."""
        M, N = matrices
        P = Matrix("P")
        expr = MAdd(M, P)
        result = _vector_from_matrix(expr, N)
        assert result.is_zero

    def test_mmmul_right_is_target(self, matrices):
        """MMMul(M, target) → coefficient is M."""
        M, N = matrices
        result = _vector_from_matrix(MMMul(M, N), N)
        assert str(result) == str(M)

    def test_mmmul_neither_has_target(self, matrices):
        """MMMul(M, P) where neither contains target returns ZeroMatrix."""
        M, N = matrices
        P = Matrix("P")
        result = _vector_from_matrix(MMMul(M, P), N)
        assert result.is_zero

    def test_smmul_no_query(self, vectors, matrices):
        """SMMul(M, s) where neither contains query → ZeroMatrix."""
        v, _, _ = vectors
        M, _ = matrices
        s = Scalar("s")
        result = _vector_from_matrix(SMMul(M, s), v)
        assert result.is_zero


# ===========================================================================
# Integration / end-to-end
# ===========================================================================


class TestExtractEndToEnd:
    """End-to-end extraction scenarios matching dynamics pipeline usage."""

    def test_simple_kinetic_energy_extraction(self, vectors, matrices):
        """Extract vec from Dot(vec, M*vec) — common in Lagrangian mechanics.

        KE = 0.5 * dot(qdot, M * qdot)
        After variation: terms like dot(delta_q, M * qdot)
        extract_linear_coeff should give M * qdot.
        """
        v, w, _ = vectors
        M, _ = matrices
        # dot(v, M*w) — extract v
        expr = Dot(v, MVMul(M, w))
        result = extract_linear_coeff(expr, v)
        # v is on the left, v == v → return right = MVMul(M, w)
        assert isinstance(result, MVMul)
        assert str(result.left) == "M"
        assert str(result.right) == "w"

    def test_cross_in_dot_extraction(self, vectors):
        """Extract vec from Dot(vec, Cross(a, b)) — common in angular momentum."""
        v, w, u = vectors
        expr = Dot(v, Cross(w, u))
        result = extract_linear_coeff(expr, v)
        assert isinstance(result, Cross)
        assert str(result.left) == "w"
        assert str(result.right) == "u"

    def test_scalar_mul_dot_extraction(self, vectors):
        """Extract vec from s * Dot(vec, w)."""
        v, w, _ = vectors
        s = Scalar("s")
        expr = Mul(s, Dot(v, w))
        result = extract_linear_coeff(expr, v)
        # _vector_from_scalar(Mul(s, Dot(v, w)), v):
        #   right has v → SVMul(_vector_from_scalar(Dot(v, w), v), s) = SVMul(w, s)
        assert isinstance(result, SVMul)
        assert str(result.left) == "w"
        assert str(result.right) == "s"

    def test_add_of_dots_extraction(self, vectors):
        """Extract vec from Dot(vec, w) + Dot(vec, u)."""
        v, w, u = vectors
        expr = Add(Dot(v, w), Dot(v, u))
        result = extract_linear_coeff(expr, v)
        assert isinstance(result, VAdd)
        assert len(result.nodes) == 2
        strs = {str(n) for n in result.nodes}
        assert strs == {"w", "u"}

    def test_mvmul_extraction_from_vector(self, vectors, matrices):
        """Extract vec from MVMul(M, vec) directly."""
        v, _, _ = vectors
        M, _ = matrices
        expr = MVMul(M, v)
        result = extract_linear_coeff(expr, v)
        assert str(result) == str(M)


# ===========================================================================
# scalar_from_scalar — Scalar query from scalar expression
# ===========================================================================


class TestScalarFromScalar:
    """scalar_from_scalar: extract scalar coefficient of scalar query."""

    def test_identity(self, scalars):
        """query == expr → 1."""
        a, _, _ = scalars
        from geomech.core.math.extract import _scalar_from_scalar

        result = _scalar_from_scalar(a, a)
        assert str(result) == "1"

    def test_mul_left(self, scalars):
        """a from m*a → m."""
        a, b, _ = scalars
        from geomech.core.math.extract import _scalar_from_scalar

        m = Scalar("m", attr=["Constant"])
        result = _scalar_from_scalar(Mul(m, a), a)
        assert str(result) == "m1"  # Mul(m, One)

    def test_mul_right(self, scalars):
        """a from a*b → b."""
        a, b, _ = scalars
        from geomech.core.math.extract import _scalar_from_scalar

        result = _scalar_from_scalar(Mul(a, b), a)
        assert str(result) == "1b"

    def test_add(self, scalars):
        """a from (a + b) → 1 (only the a term)."""
        a, b, _ = scalars
        from geomech.core.math.extract import _scalar_from_scalar

        result = _scalar_from_scalar(Add(a, b), a)
        assert str(result) == "1"

    def test_not_present(self, scalars):
        """a from b → 0."""
        a, b, _ = scalars
        from geomech.core.math.extract import _scalar_from_scalar

        result = _scalar_from_scalar(b, a)
        assert result.is_zero

    def test_dot_containing_scalar(self, scalars, vectors):
        """a from Dot(a*x, y) → Dot(x, y)."""
        a, _, _ = scalars
        v, w, _ = vectors
        from geomech.core.math.extract import _scalar_from_scalar

        expr = Dot(SVMul(v, a), w)
        result = _scalar_from_scalar(expr, a)
        assert isinstance(result, Dot)

    def test_nested_mul(self, scalars):
        """a from m*k*a → m*k."""
        a, _, _ = scalars
        from geomech.core.math.extract import _scalar_from_scalar

        m = Scalar("m", attr=["Constant"])
        k = Scalar("k", attr=["Constant"])
        expr = Mul(Mul(m, k), a)
        result = _scalar_from_scalar(expr, a)
        assert "m" in str(result)
        assert "k" in str(result)

    def test_vvmul_query_in_right(self, scalars, vectors):
        """a from VVMul(Transpose(x), a*y) → VVMul(Transpose(x), y)."""
        a, _, _ = scalars
        v, w, _ = vectors
        from geomech.core.math.extract import _scalar_from_scalar

        expr = VVMul(Transpose(v), SVMul(w, a))
        result = _scalar_from_scalar(expr, a)
        assert isinstance(result, VVMul)
        assert str(result.right) == "w"

    def test_vvmul_query_in_left(self, scalars, vectors):
        """a from VVMul(Transpose(a*x), y) → VVMul(Transpose(x), y)."""
        a, _, _ = scalars
        v, w, _ = vectors
        from geomech.core.math.extract import _scalar_from_scalar

        expr = VVMul(Transpose(SVMul(v, a)), w)
        result = _scalar_from_scalar(expr, a)
        assert isinstance(result, VVMul)
        assert isinstance(result.left, Transpose)

    def test_vvmul_not_present(self, scalars, vectors):
        """a from VVMul(Transpose(x), y) where neither has a → Zero."""
        a, _, _ = scalars
        v, w, _ = vectors
        from geomech.core.math.extract import _scalar_from_scalar

        expr = VVMul(Transpose(v), w)
        result = _scalar_from_scalar(expr, a)
        assert result.is_zero

    def test_vvmul_nonlinear_raises(self, scalars, vectors):
        """a in both sides of VVMul → nonlinear error."""
        a, _, _ = scalars
        v, w, _ = vectors
        from geomech.core.math.extract import _scalar_from_scalar

        expr = VVMul(Transpose(SVMul(v, a)), SVMul(w, a))
        with pytest.raises(NonLinearError, match="nonlinear"):
            _scalar_from_scalar(expr, a)


# ===========================================================================
# scalar_from_vector — Scalar query from vector expression
# ===========================================================================


class TestScalarFromVector:
    """scalar_from_vector: extract vector coefficient of scalar query."""

    def test_svmul_scalar_is_query(self, scalars, vectors):
        """a from x*a → x."""
        a, _, _ = scalars
        v, _, _ = vectors
        from geomech.core.math.extract import _scalar_from_vector

        result = _scalar_from_vector(SVMul(v, a), a)
        assert str(result) == "v"

    def test_svmul_not_present(self, scalars, vectors):
        """a from x*b → 0v."""
        a, b, _ = scalars
        v, _, _ = vectors
        from geomech.core.math.extract import _scalar_from_vector

        result = _scalar_from_vector(SVMul(v, b), a)
        assert result.is_zero

    def test_mvmul_not_present(self, scalars, vectors, matrices):
        """a from M*x → 0v."""
        a, _, _ = scalars
        v, _, _ = vectors
        M, _ = matrices
        from geomech.core.math.extract import _scalar_from_vector

        result = _scalar_from_vector(MVMul(M, v), a)
        assert result.is_zero

    def test_cross_with_scalar_in_left(self, scalars, vectors):
        """a from Cross(a*x, y) → -Hat(y)*x."""
        a, _, _ = scalars
        v, w, _ = vectors
        from geomech.core.math.extract import _scalar_from_vector

        expr = Cross(SVMul(v, a), w)
        result = _scalar_from_vector(expr, a)
        assert isinstance(result, MVMul)

    def test_vadd(self, scalars, vectors):
        """a from VAdd(x*a, y) → x."""
        a, _, _ = scalars
        v, w, _ = vectors
        from geomech.core.math.extract import _scalar_from_vector

        expr = VAdd(SVMul(v, a), w)
        result = _scalar_from_vector(expr, a)
        assert str(result) == "v"


# ===========================================================================
# scalar_from_matrix — Scalar query from matrix expression
# ===========================================================================


class TestScalarFromMatrix:
    """scalar_from_matrix: extract matrix coefficient of scalar query."""

    def test_smmul_scalar_is_query(self, scalars, matrices):
        """a from M*a → M."""
        a, _, _ = scalars
        M, _ = matrices
        from geomech.core.math.extract import _scalar_from_matrix

        result = _scalar_from_matrix(SMMul(M, a), a)
        assert str(result) == "M"

    def test_not_present(self, scalars, matrices):
        """a from M*N → 0."""
        a, _, _ = scalars
        M, N = matrices
        from geomech.core.math.extract import _scalar_from_matrix

        result = _scalar_from_matrix(MMMul(M, N), a)
        assert result.is_zero

    def test_mmmul_with_scalar(self, scalars, matrices):
        """a from MMMul(SMMul(M, a), N) → MMMul(M, N)."""
        a, _, _ = scalars
        M, N = matrices
        from geomech.core.math.extract import _scalar_from_matrix

        expr = MMMul(SMMul(M, a), N)
        result = _scalar_from_matrix(expr, a)
        assert isinstance(result, MMMul)
        assert str(result.left) == "M"
        assert str(result.right) == "N"


# ===========================================================================
# matrix_from_matrix — Matrix query from matrix expression
# ===========================================================================


class TestMatrixFromMatrix:
    """matrix_from_matrix: extract matrix coefficient of matrix query."""

    def test_identity(self, matrices):
        """N from N → I."""
        _, N = matrices
        from geomech.core.math.extract import _matrix_from_matrix

        result = _matrix_from_matrix(N, N)
        assert str(result) == "I"

    def test_mmmul_right(self, matrices):
        """N from M*N → M."""
        M, N = matrices
        from geomech.core.math.extract import _matrix_from_matrix

        result = _matrix_from_matrix(MMMul(M, N), N)
        assert str(result) == "M"

    def test_mmmul_left(self, matrices):
        """M from M*N → N."""
        M, N = matrices
        from geomech.core.math.extract import _matrix_from_matrix

        result = _matrix_from_matrix(MMMul(M, N), M)
        assert str(result) == "N"

    def test_not_present(self, matrices):
        """N from M*P → 0."""
        M, N = matrices
        P = Matrix("P")
        from geomech.core.math.extract import _matrix_from_matrix

        result = _matrix_from_matrix(MMMul(M, P), N)
        assert result.is_zero

    def test_smmul(self, scalars, matrices):
        """N from SMMul(N, s) → SMMul(I, s)."""
        _, N = matrices
        from geomech.core.math.extract import _matrix_from_matrix

        s = Scalar("s")
        result = _matrix_from_matrix(SMMul(N, s), N)
        assert str(result) == "Is"


# ===========================================================================
# extract_linear_coeff dispatch — scalar queries
# ===========================================================================


class TestExtractCoeffScalarDispatch:
    """extract_linear_coeff dispatches correctly for scalar queries."""

    def test_scalar_from_scalar_via_dispatch(self, scalars):
        """extract_linear_coeff(m*a, a) → m (scalar from scalar)."""
        a, _, _ = scalars
        m = Scalar("m", attr=["Constant"])
        result = extract_linear_coeff(Mul(m, a), a)
        assert "m" in str(result)

    def test_scalar_from_vector_via_dispatch(self, scalars, vectors):
        """extract_linear_coeff(x*a, a) → x (scalar from vector)."""
        a, _, _ = scalars
        v, _, _ = vectors
        result = extract_linear_coeff(SVMul(v, a), a)
        assert str(result) == "v"

    def test_scalar_from_dot_via_dispatch(self, scalars, vectors):
        """extract_linear_coeff(Dot(a*x, y), a) dispatches to scalar_from_scalar."""
        a, _, _ = scalars
        v, w, _ = vectors
        result = extract_linear_coeff(Dot(SVMul(v, a), w), a)
        assert isinstance(result, Dot)


# ===========================================================================
# NotImplementedError coverage
# ===========================================================================


class TestNotImplementedErrors:
    """Verify all NotImplementedError paths raise with correct messages."""

    # --- _scalar_from_scalar nonlinear ---

    def test_scalar_from_scalar_mul_nonlinear(self, scalars):
        """a*a is nonlinear in Mul."""
        a, _, _ = scalars
        from geomech.core.math.extract import _scalar_from_scalar

        with pytest.raises(NonLinearError, match="nonlinear"):
            _scalar_from_scalar(Mul(a, a), a)

    def test_scalar_from_scalar_dot_nonlinear(self, scalars, vectors):
        """Dot(a*v, a*w) where scalar a appears in both sides."""
        a, _, _ = scalars
        v, w, _ = vectors
        from geomech.core.math.extract import _scalar_from_scalar

        expr = Dot(SVMul(v, a), SVMul(w, a))
        with pytest.raises(NonLinearError, match="nonlinear"):
            _scalar_from_scalar(expr, a)

    def test_scalar_from_scalar_vvmul_nonlinear(self, scalars, vectors):
        """VVMul with query in both sides."""
        a, _, _ = scalars
        v, w, _ = vectors
        from geomech.core.math.extract import _scalar_from_scalar

        expr = VVMul(Transpose(SVMul(v, a)), SVMul(w, a))
        with pytest.raises(NonLinearError, match="nonlinear"):
            _scalar_from_scalar(expr, a)

    # --- _scalar_from_vector nonlinear ---

    def test_scalar_from_vector_svmul_nonlinear(self, scalars, vectors):
        """SVMul with query in both vec and scl."""
        a, _, _ = scalars
        v, _, _ = vectors
        from geomech.core.math.extract import _scalar_from_vector

        expr = SVMul(SVMul(v, a), a)
        with pytest.raises(NonLinearError, match="nonlinear"):
            _scalar_from_vector(expr, a)

    def test_scalar_from_vector_mvmul_nonlinear(self, scalars, vectors, matrices):
        """MVMul with query in both mat and vec."""
        a, _, _ = scalars
        v, _, _ = vectors
        M, _ = matrices
        from geomech.core.math.extract import _scalar_from_vector

        expr = MVMul(SMMul(M, a), SVMul(v, a))
        with pytest.raises(NonLinearError, match="nonlinear"):
            _scalar_from_vector(expr, a)

    def test_scalar_from_vector_cross_nonlinear(self, scalars, vectors):
        """Cross with query in both sides."""
        a, _, _ = scalars
        v, w, _ = vectors
        from geomech.core.math.extract import _scalar_from_vector

        expr = Cross(SVMul(v, a), SVMul(w, a))
        with pytest.raises(NonLinearError, match="nonlinear"):
            _scalar_from_vector(expr, a)

    # --- _scalar_from_matrix nonlinear ---

    def test_scalar_from_matrix_smmul_nonlinear(self, scalars, matrices):
        """SMMul with query in both mat and scl."""
        a, _, _ = scalars
        M, _ = matrices
        from geomech.core.math.extract import _scalar_from_matrix

        expr = SMMul(SMMul(M, a), a)
        with pytest.raises(NonLinearError, match="nonlinear"):
            _scalar_from_matrix(expr, a)

    def test_scalar_from_matrix_mmmul_nonlinear(self, scalars, matrices):
        """MMMul with query in both sides."""
        a, _, _ = scalars
        M, N = matrices
        from geomech.core.math.extract import _scalar_from_matrix

        expr = MMMul(SMMul(M, a), SMMul(N, a))
        with pytest.raises(NonLinearError, match="nonlinear"):
            _scalar_from_matrix(expr, a)

    # --- _vector_from_scalar ---

    def test_vector_from_scalar_mul_nonlinear(self, vectors):
        """Mul with query in both sides."""
        v, w, _ = vectors
        expr = Mul(Dot(v, w), Dot(w, v))
        with pytest.raises(NonLinearError, match="nonlinear"):
            _vector_from_scalar(expr, v)

    def test_vector_from_scalar_dot_nonlinear(self, vectors):
        """Dot with query in both sides."""
        v, w, u = vectors
        expr = Dot(Cross(w, v), Cross(v, u))
        with pytest.raises(NonLinearError, match="nonlinear"):
            _vector_from_scalar(expr, v)

    def test_vector_from_scalar_vvmul_left(self, vectors):
        """VVMul(Transpose(v), w) extract v → w."""
        v, w, _ = vectors
        expr = VVMul(Transpose(v), w)
        result = _vector_from_scalar(expr, v)
        assert str(result) == "w"

    def test_vector_from_scalar_vvmul_right(self, vectors):
        """VVMul(Transpose(w), v) extract v → w."""
        v, w, _ = vectors
        expr = VVMul(Transpose(w), v)
        result = _vector_from_scalar(expr, v)
        assert str(result) == "w"

    def test_vector_from_scalar_vvmul_nested(self, vectors, matrices):
        """VVMul(Transpose(M*v), w) extract v → M^T * w."""
        v, w, _ = vectors
        M, _ = matrices
        expr = VVMul(Transpose(MVMul(M, v)), w)
        result = _vector_from_scalar(expr, v)
        assert isinstance(result, MVMul)

    def test_vector_from_scalar_vvmul_nonlinear(self, vectors):
        """VVMul with query in both sides raises."""
        v, w, _ = vectors
        expr = VVMul(Transpose(Cross(w, v)), Cross(w, v))
        with pytest.raises(NonLinearError, match="nonlinear"):
            _vector_from_scalar(expr, v)

    # --- _vector_from_vector nonlinear ---

    def test_vector_from_vector_cross_nonlinear(self, vectors, matrices):
        """Cross with query in both sides."""
        v, _, _ = vectors
        M, N = matrices
        expr = Cross(MVMul(M, v), MVMul(N, v))
        with pytest.raises(NonLinearError, match="nonlinear"):
            _vector_from_vector(expr, v)

    def test_vector_from_vector_svmul_nonlinear(self, vectors, matrices):
        """SVMul with query in both vec and scl."""
        v, w, _ = vectors
        M, _ = matrices
        expr = SVMul(MVMul(M, v), Dot(v, w))
        with pytest.raises(NonLinearError, match="nonlinear"):
            _vector_from_vector(expr, v)

    # --- _vector_from_mvmul_mat ---

    def test_mvmul_mat_smmul_scalar_has_query(self, vectors):
        """MVMul(SMMul(Hat(w), Dot(v,w)), u) extract v → redistributes through SVMul scalar."""
        v, w, u = vectors
        # SMMul(Hat(w), Dot(v,w)) * u = Hat(w) * (Dot(v,w) * u)
        # → query v in Dot scalar → MVMul(VVMul(u, Tr(w)), v) → coeff = VVMul(u, Tr(w))
        # then Hat(w) * coeff
        expr = MVMul(SMMul(Hat(w), Dot(v, w)), u)
        result = _vector_from_vector(expr, v)
        assert result is not ZeroMatrix

    def test_mvmul_mat_vvmul(self, vectors):
        """MVMul(VVMul(v, Tr(w)), u) extract v → rewrites to SVMul(v, Dot(w, u))."""
        v, w, u = vectors
        mat = VVMul(v, Transpose(w))  # outer product → matrix
        expr = MVMul(mat, u)
        # (v * w^T) * u = v * (w^T * u) = SVMul(v, Dot(w, u))
        # v is the query leaf → coeff = SMMul(I, Dot(w, u)) or just Dot(w, u) scaled
        result = _vector_from_vector(expr, v)
        assert result is not ZeroMatrix

    # --- _vector_from_vector_svmul_scalar ---

    def test_svmul_scalar_mul_nonlinear(self, vectors):
        """SVMul(w, Mul(Dot(v,u), Dot(v,w))) — query in both sides of inner Mul."""
        v, w, u = vectors
        scl = Mul(Dot(v, u), Dot(v, w))
        expr = SVMul(w, scl)
        with pytest.raises(NonLinearError, match="nonlinear"):
            _vector_from_vector(expr, v)

    def test_svmul_scalar_dot_nonlinear(self, vectors):
        """SVMul(w, Dot(Cross(u,v), Cross(w,v))) — query in both sides of inner Dot."""
        v, w, u = vectors
        scl = Dot(Cross(u, v), Cross(w, v))
        expr = SVMul(w, scl)
        with pytest.raises(NonLinearError, match="nonlinear"):
            _vector_from_vector(expr, v)

    def test_svmul_scalar_vvmul_redistribution(self, vectors):
        """SVMul(w, VVMul(Transpose(v), u)) extract v → redistributes outer product."""
        v, w, u = vectors
        scl = VVMul(Transpose(v), u)
        expr = SVMul(w, scl)
        result = _vector_from_vector(expr, v)
        # VVMul(Tr(v), u) = v^T * u, query in left.expr → MVMul(VVMul(w, Tr(u)), v)
        # v is leaf query → coeff = VVMul(w, Transpose(u))
        assert isinstance(result, VVMul)

    # --- _vector_from_matrix ---

    def test_vector_from_matrix_mmmul_nonlinear(self, vectors, matrices):
        """MMMul with query in both sides."""
        v, _, _ = vectors
        from geomech.core.math.extract import _vector_from_matrix

        expr = MMMul(Hat(v), Hat(v))
        with pytest.raises(NonLinearError, match="nonlinear"):
            _vector_from_matrix(expr, v)

    def test_vector_from_matrix_mmmul_left(self, vectors, matrices):
        """MMMul(Hat(v), M) extract v → MMMul(coeff, M)."""
        v, _, _ = vectors
        M, _ = matrices
        from geomech.core.math.extract import _vector_from_matrix

        expr = MMMul(Hat(v), M)
        result = _vector_from_matrix(expr, v)
        assert isinstance(result, MMMul)
        assert str(result.right) == "M"

    def test_vector_from_matrix_mmmul_left_is_query(self, vectors, matrices):
        """MMMul(Hat(v), M) where Hat(v) is the query itself — unlikely but covers leaf."""
        v, _, _ = vectors
        M, _ = matrices
        from geomech.core.math.extract import _vector_from_matrix

        expr = MMMul(Hat(v), M)
        result = _vector_from_matrix(expr, Hat(v))
        assert str(result) == "M"

    def test_vector_from_matrix_mmmul_right_nonleaf(self, vectors, matrices):
        """MMMul(M, MMMul(Hat(v), M)) extract v → MMMul(M, coeff)."""
        v, w, _ = vectors
        M, _ = matrices
        from geomech.core.math.extract import _vector_from_matrix

        expr = MMMul(M, MMMul(Hat(v), M))
        result = _vector_from_matrix(expr, v)
        assert isinstance(result, MMMul)
        assert str(result.left) == "M"

    def test_vector_from_matrix_vvmul(self, vectors):
        """VVMul outer product cannot be factored as C * q — AlgebraicError."""
        v, w, _ = vectors
        from geomech.core.math.extract import _vector_from_matrix

        expr = VVMul(v, Transpose(w))
        with pytest.raises(AlgebraicError, match="outer product"):
            _vector_from_matrix(expr, v)

    def test_vector_from_matrix_smmul_mat_side(self, vectors, matrices):
        """SMMul(Hat(v), s) extract v → SMMul(coeff, s)."""
        v, _, _ = vectors
        from geomech.core.math.extract import _vector_from_matrix

        s = Scalar("s")
        expr = SMMul(Hat(v), s)
        result = _vector_from_matrix(expr, v)
        assert isinstance(result, SMMul)
        assert str(result.right) == "s"

    def test_vector_from_matrix_smmul_scalar_side(self, vectors, matrices):
        """SMMul(M, Dot(v,w)) — scalar depends on query, AlgebraicError."""
        v, w, _ = vectors
        M, _ = matrices
        from geomech.core.math.extract import _vector_from_matrix

        expr = SMMul(M, Dot(v, w))
        with pytest.raises(AlgebraicError, match="scalar factor depends on query"):
            _vector_from_matrix(expr, v)

    def test_vector_from_matrix_smmul_nonlinear(self, vectors, matrices):
        """SMMul with query in both sides."""
        v, w, _ = vectors
        from geomech.core.math.extract import _vector_from_matrix

        expr = SMMul(Hat(v), Dot(v, w))
        with pytest.raises(NonLinearError, match="nonlinear"):
            _vector_from_matrix(expr, v)

    # --- _matrix_from_matrix nonlinear ---

    def test_matrix_from_matrix_mmmul_nonlinear(self, matrices):
        """MMMul with query in both sides."""
        M, N = matrices
        from geomech.core.math.extract import _matrix_from_matrix

        expr = MMMul(MMMul(M, N), MMMul(N, M))
        with pytest.raises(NonLinearError, match="nonlinear"):
            _matrix_from_matrix(expr, N)
