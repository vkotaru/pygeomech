"""Tests for geomech.core.math.extract — coefficient extraction."""

import pytest

from geomech.core.base.expressions import (
    Scalar, Vector, Matrix, Zero, ZeroVector, ZeroMatrix,
)
from geomech.core.operations.addition import Add, VAdd, MAdd
from geomech.core.operations.multiplication import (
    Mul, SVMul, SMMul, MVMul, MMMul,
)
from geomech.core.operations.geometry import Dot, Cross, Hat, Transpose
from geomech.core.math.extract import (
    extract_coeff, extract_from_scalar, extract_from_vector, extract_from_matrix,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def scalars():
    a = Scalar('a')
    b = Scalar('b')
    c = Scalar('c', value=3, attr=['Constant'])
    return a, b, c


@pytest.fixture
def vectors():
    v = Vector('v')
    w = Vector('w')
    u = Vector('u')
    return v, w, u


@pytest.fixture
def matrices():
    M = Matrix('M')
    N = Matrix('N')
    return M, N


# ===========================================================================
# extract_coeff dispatch
# ===========================================================================

class TestExtractCoeffDispatch:
    """extract_coeff routes to extract_from_scalar/extract_from_vector/extract_from_matrix based on expr.type."""

    def test_scalar_dispatches_to_extract_from_scalar(self, vectors):
        """Scalar expression dispatches to extract_from_scalar."""
        v, w, _ = vectors
        expr = Dot(v, w)
        result = extract_coeff(expr, v)
        assert str(result) == str(w)

    def test_vector_dispatches_to_extract_from_vector(self, vectors, matrices):
        """Vector expression dispatches to extract_from_vector."""
        v, _, _ = vectors
        M, _ = matrices
        expr = MVMul(M, v)
        result = extract_coeff(expr, v)
        assert str(result) == str(M)

    def test_matrix_dispatches_to_extract_from_matrix(self, matrices):
        """Matrix expression dispatches to extract_from_matrix."""
        M, N = matrices
        expr = MMMul(M, N)
        result = extract_coeff(expr, N)
        assert str(result) == str(M)


# ===========================================================================
# extract_from_scalar — Extract From Scalar
# ===========================================================================

class TestExtractFromScalarAdd:
    """extract_from_scalar through scalar addition."""

    def test_single_dot_in_add(self, vectors):
        """Add with one term containing vec extracts that term only."""
        v, w, u = vectors
        a = Scalar('a')
        # Dot(v, w) + a  →  extract_from_scalar extracts from Dot(v, w) only
        expr = Add(Dot(v, w), a)
        result = extract_from_scalar(expr, v)
        assert str(result) == str(w)

    def test_multiple_dots_in_add(self, vectors):
        """Add with multiple terms containing vec combines via VAdd."""
        v, w, u = vectors
        expr = Add(Dot(v, w), Dot(v, u))
        result = extract_from_scalar(expr, v)
        assert isinstance(result, VAdd)
        assert len(result.nodes) == 2

    def test_no_terms_with_vec(self, vectors, scalars):
        """Add where no term contains vec returns ZeroVector."""
        v, w, u = vectors
        a, b, _ = scalars
        expr = Add(a, b)
        result = extract_from_scalar(expr, v)
        assert result.is_zero


class TestExtractFromScalarMul:
    """extract_from_scalar through scalar multiplication."""

    def test_vec_in_left(self, vectors):
        """Mul(dot(v,w), s) → SVMul(extract_from_scalar(dot(v,w), v), s) = SVMul(w, s)."""
        v, w, _ = vectors
        s = Scalar('s')
        expr = Mul(Dot(v, w), s)
        result = extract_from_scalar(expr, v)
        assert isinstance(result, SVMul)
        assert str(result.right) == 's'

    def test_vec_in_right(self, vectors):
        """Mul(s, dot(v,w)) → SVMul(extract_from_scalar(dot(v,w), v), s) = SVMul(w, s)."""
        v, w, _ = vectors
        s = Scalar('s')
        expr = Mul(s, Dot(v, w))
        result = extract_from_scalar(expr, v)
        assert isinstance(result, SVMul)
        assert str(result.right) == 's'

    def test_neither_side_has_vec(self, vectors, scalars):
        """Mul(a, b) where neither contains vec returns ZeroVector."""
        v, _, _ = vectors
        a, b, _ = scalars
        expr = Mul(a, b)
        result = extract_from_scalar(expr, v)
        assert result.is_zero


class TestExtractFromScalarDot:
    """extract_from_scalar through dot product — the core extraction case."""

    def test_left_is_vec(self, vectors):
        """Dot(vec, w) → coefficient is w."""
        v, w, _ = vectors
        expr = Dot(v, w)
        result = extract_from_scalar(expr, v)
        assert str(result) == str(w)

    def test_right_is_vec(self, vectors):
        """Dot(w, vec) → coefficient is w."""
        v, w, _ = vectors
        expr = Dot(w, v)
        result = extract_from_scalar(expr, v)
        assert str(result) == str(w)

    def test_left_is_cross_containing_vec(self, vectors):
        """Dot(Cross(a, vec), b) → MVMul(Transpose(extract_from_vector(Cross(a,vec), vec)), b).

        extract_from_vector(Cross(a, vec), vec) = Hat(a), so result = Transpose(Hat(a)) * b.
        """
        v, w, u = vectors
        expr = Dot(Cross(w, v), u)
        result = extract_from_scalar(expr, v)
        # extract_from_vector(Cross(w, v), v) = Hat(w), so result = MVMul(Transpose(Hat(w)), u)
        assert isinstance(result, MVMul)
        assert isinstance(result.left, Transpose)

    def test_right_is_cross_containing_vec(self, vectors):
        """Dot(b, Cross(a, vec)) → MVMul(Transpose(extract_from_vector(Cross(a,vec), vec)), b)."""
        v, w, u = vectors
        expr = Dot(u, Cross(w, v))
        result = extract_from_scalar(expr, v)
        assert isinstance(result, MVMul)
        assert isinstance(result.left, Transpose)

    def test_neither_side_has_vec(self, vectors):
        """Dot(a, b) where neither contains vec returns ZeroVector."""
        v, w, u = vectors
        expr = Dot(w, u)
        result = extract_from_scalar(expr, v)
        assert result.is_zero

    def test_both_sides_have_vec_raises(self, vectors):
        """Dot(f(vec), g(vec)) raises NotImplementedError."""
        v, w, _ = vectors
        expr = Dot(Cross(w, v), Cross(v, w))
        with pytest.raises(NotImplementedError, match="both sides"):
            extract_from_scalar(expr, v)


class TestExtractFromScalarLeaf:
    """extract_from_scalar on leaf / unhandled expressions."""

    def test_plain_scalar_returns_zero(self, vectors, scalars):
        """Plain scalar leaf returns ZeroVector."""
        v, _, _ = vectors
        a, _, _ = scalars
        result = extract_from_scalar(a, v)
        assert result.is_zero


# ===========================================================================
# extract_from_vector — Extract From Vector
# ===========================================================================

class TestExtractFromVectorVAdd:
    """extract_from_vector through vector addition."""

    def test_single_term_with_vec(self, vectors, matrices):
        """VAdd with one term containing vec extracts that term."""
        v, w, _ = vectors
        M, _ = matrices
        expr = VAdd(MVMul(M, v), w)
        result = extract_from_vector(expr, v)
        assert str(result) == str(M)

    def test_multiple_terms_with_vec(self, vectors, matrices):
        """VAdd with multiple terms containing vec combines via MAdd."""
        v, _, _ = vectors
        M, N = matrices
        expr = VAdd(MVMul(M, v), MVMul(N, v))
        result = extract_from_vector(expr, v)
        assert isinstance(result, MAdd)
        assert len(result.nodes) == 2

    def test_no_terms_with_vec(self, vectors, matrices):
        """VAdd where no term contains vec returns ZeroMatrix."""
        v, w, u = vectors
        expr = VAdd(w, u)
        result = extract_from_vector(expr, v)
        assert result.is_zero


class TestExtractFromVectorCross:
    """extract_from_vector through cross product."""

    def test_cross_right_is_vec(self, vectors):
        """Cross(a, vec) = Hat(a) * vec → coefficient is Hat(a)."""
        v, w, _ = vectors
        result = extract_from_vector(Cross(w, v), v)
        assert isinstance(result, Hat)
        assert str(result.expr) == 'w'

    def test_cross_left_is_vec(self, vectors):
        """Cross(vec, b) = -Hat(b) * vec → coefficient is -Hat(b)."""
        v, w, _ = vectors
        result = extract_from_vector(Cross(v, w), v)
        assert isinstance(result, SMMul)
        assert isinstance(result.left, Hat)
        assert str(result.left.expr) == 'w'

    def test_cross_self(self, vectors):
        """Cross(v, v) = 0 → coefficient is ZeroMatrix."""
        v, _, _ = vectors
        result = extract_from_vector(Cross(v, v), v)
        assert result.is_zero

    def test_cross_left_contains_vec_recursive(self, vectors, matrices):
        """Cross(M*vec, b) → -Hat(b) * extract_from_vector(M*vec, vec) = -Hat(b) * M."""
        v, w, _ = vectors
        M, _ = matrices
        expr = Cross(MVMul(M, v), w)
        result = extract_from_vector(expr, v)
        assert isinstance(result, MMMul)
        # left should be SMMul(Hat(w), -1), right should be M
        assert isinstance(result.left, SMMul)
        assert str(result.right) == str(M)

    def test_cross_right_contains_vec_recursive(self, vectors, matrices):
        """Cross(a, M*vec) → Hat(a) * extract_from_vector(M*vec, vec) = Hat(a) * M."""
        v, w, _ = vectors
        M, _ = matrices
        expr = Cross(w, MVMul(M, v))
        result = extract_from_vector(expr, v)
        assert isinstance(result, MMMul)
        assert isinstance(result.left, Hat)
        assert str(result.right) == str(M)

    def test_cross_neither_has_vec(self, vectors):
        """Cross(a, b) where neither is vec returns ZeroMatrix."""
        v, w, u = vectors
        result = extract_from_vector(Cross(w, u), v)
        assert result.is_zero


class TestExtractFromVectorMVMul:
    """extract_from_vector through matrix-vector multiplication."""

    def test_right_is_vec(self, vectors, matrices):
        """MVMul(M, vec) → coefficient is M."""
        v, _, _ = vectors
        M, _ = matrices
        result = extract_from_vector(MVMul(M, v), v)
        assert str(result) == str(M)

    def test_right_contains_vec(self, vectors, matrices):
        """MVMul(M, Cross(a, vec)) → MMMul(M, extract_from_vector(Cross(a,vec), vec)) = MMMul(M, Hat(a))."""
        v, w, _ = vectors
        M, _ = matrices
        expr = MVMul(M, Cross(w, v))
        result = extract_from_vector(expr, v)
        assert isinstance(result, MMMul)
        assert str(result.left) == str(M)
        assert isinstance(result.right, Hat)

    def test_neither_side_has_vec(self, vectors, matrices):
        """MVMul(M, w) where w is not vec returns ZeroMatrix."""
        v, w, _ = vectors
        M, _ = matrices
        result = extract_from_vector(MVMul(M, w), v)
        assert result.is_zero


class TestExtractFromVectorMVMulMatSide:
    """extract_from_vector through MVMul where the matrix side contains vec."""

    def test_hat_of_vec(self, vectors):
        """MVMul(Hat(vec), b) = cross(vec, b) → -Hat(b)."""
        v, w, _ = vectors
        expr = MVMul(Hat(v), w)
        result = extract_from_vector(expr, v)
        assert isinstance(result, SMMul)
        assert isinstance(result.left, Hat)
        assert str(result.left.expr) == 'w'

    def test_hat_of_expr_containing_vec(self, vectors, matrices):
        """MVMul(Hat(M*vec), b) → -Hat(b) * extract_from_vector(M*vec, vec) = -Hat(b) * M."""
        v, w, _ = vectors
        M, _ = matrices
        expr = MVMul(Hat(MVMul(M, v)), w)
        result = extract_from_vector(expr, v)
        assert isinstance(result, MMMul)
        assert isinstance(result.left, SMMul)
        assert str(result.right) == str(M)

    def test_mmmul_right_has_vec(self, vectors, matrices):
        """MVMul(MMMul(A, B), b) where B has vec → A * extract_from_vector(MVMul(B, b), vec)."""
        v, w, _ = vectors
        M, N = matrices
        # MVMul(MMMul(M, Hat(v)), w) — Hat(v) contains v
        # → MMMul(M, extract_from_vector(MVMul(Hat(v), w), v))
        # → MMMul(M, SMMul(Hat(w), -1))
        expr = MVMul(MMMul(M, Hat(v)), w)
        result = extract_from_vector(expr, v)
        assert isinstance(result, MMMul)
        assert str(result.left) == str(M)

    def test_smmul_mat_has_vec(self, vectors):
        """MVMul(SMMul(Hat(vec), s), b) redistributes scalar then extracts.

        SMMul(Hat(v), s) * w → MVMul(Hat(v), SVMul(w, s))
        → Hat(v) contains v → _extract_from_vector_mvmul_mat → Hat case: expr == v → SMMul(Hat(SVMul(w,s)), -1)
        """
        v, w, _ = vectors
        s = Scalar('s')
        expr = MVMul(SMMul(Hat(v), s), w)
        result = extract_from_vector(expr, v)
        assert isinstance(result, SMMul)
        assert isinstance(result.left, Hat)


class TestExtractFromVectorMVMulMAddMat:
    """extract_from_vector through MVMul where the matrix side is an MAdd."""

    def test_madd_one_term_has_vec(self, vectors, matrices):
        """MVMul(MAdd(Hat(vec), N), b) → extracts from the Hat(vec) term only."""
        v, w, _ = vectors
        _, N = matrices
        expr = MVMul(MAdd(Hat(v), N), w)
        result = extract_from_vector(expr, v)
        # Hat(v) has v, N doesn't → extract_from_vector(MVMul(Hat(v), w), v) → SMMul(Hat(w), -1)
        assert isinstance(result, SMMul)
        assert isinstance(result.left, Hat)

    def test_madd_both_terms_have_vec(self, vectors):
        """MVMul(MAdd(Hat(v), SMMul(Hat(v),-1)), w) → MAdd of two coefficients."""
        v, w, _ = vectors
        mat = MAdd(Hat(v), SMMul(Hat(v), -1))
        expr = MVMul(mat, w)
        result = extract_from_vector(expr, v)
        assert isinstance(result, MAdd)
        assert len(result.nodes) == 2


class TestExtractFromVectorLeaf:
    """extract_from_vector on leaf expressions."""

    def test_plain_vector_not_target_raises(self, vectors):
        """extract_from_vector on a plain Vector that isn't the target raises NotImplementedError."""
        v, w, _ = vectors
        with pytest.raises(NotImplementedError, match="unhandled"):
            extract_from_vector(w, v)


class TestExtractFromVectorSVMul:
    """extract_from_vector through scalar-vector multiplication."""

    def test_vec_side_contains_vec(self, vectors, matrices):
        """SVMul(M*vec, s) → SMMul(extract_from_vector(M*vec, vec), s) = SMMul(M, s)."""
        v, _, _ = vectors
        M, _ = matrices
        s = Scalar('s')
        expr = SVMul(MVMul(M, v), s)
        result = extract_from_vector(expr, v)
        assert isinstance(result, SMMul)
        assert str(result.right) == 's'

    def test_scalar_side_contains_vec_raises(self, vectors):
        """SVMul(w, dot(v, u)) raises NotImplementedError for scalar side."""
        v, w, u = vectors
        expr = SVMul(w, Dot(v, u))
        with pytest.raises(NotImplementedError, match="scalar contains vec"):
            extract_from_vector(expr, v)


# ===========================================================================
# extract_from_matrix — Extract From Matrix
# ===========================================================================

class TestExtractFromMatrix:
    """extract_from_matrix — extract from matrix."""

    def test_madd_one_matching_term(self, matrices):
        """MAdd with one term containing target extracts it."""
        M, N = matrices
        P = Matrix('P')
        expr = MAdd(MMMul(M, N), P)
        result = extract_from_matrix(expr, N)
        assert str(result) == str(M)

    def test_madd_multiple_matching_terms(self, matrices):
        """MAdd with multiple terms containing target combines via MAdd."""
        M, N = matrices
        P = Matrix('P')
        expr = MAdd(MMMul(M, N), MMMul(P, N))
        result = extract_from_matrix(expr, N)
        assert isinstance(result, MAdd)
        assert len(result.nodes) == 2

    def test_madd_no_matching_terms(self, matrices):
        """MAdd where no term contains target returns ZeroMatrix."""
        M, N = matrices
        P = Matrix('P')
        expr = MAdd(M, P)
        result = extract_from_matrix(expr, N)
        assert result.is_zero

    def test_mmmul_right_is_target(self, matrices):
        """MMMul(M, target) → coefficient is M."""
        M, N = matrices
        result = extract_from_matrix(MMMul(M, N), N)
        assert str(result) == str(M)

    def test_mmmul_neither_has_target(self, matrices):
        """MMMul(M, P) where neither contains target returns ZeroMatrix."""
        M, N = matrices
        P = Matrix('P')
        result = extract_from_matrix(MMMul(M, P), N)
        assert result.is_zero

    def test_smmul_raises(self, vectors, matrices):
        """SMMul raises NotImplementedError."""
        v, _, _ = vectors
        M, _ = matrices
        s = Scalar('s')
        with pytest.raises(NotImplementedError, match="SMMul"):
            extract_from_matrix(SMMul(M, s), v)


# ===========================================================================
# Integration / end-to-end
# ===========================================================================

class TestExtractEndToEnd:
    """End-to-end extraction scenarios matching dynamics pipeline usage."""

    def test_simple_kinetic_energy_extraction(self, vectors, matrices):
        """Extract vec from Dot(vec, M*vec) — common in Lagrangian mechanics.

        KE = 0.5 * dot(qdot, M * qdot)
        After variation: terms like dot(delta_q, M * qdot)
        extract_coeff should give M * qdot.
        """
        v, w, _ = vectors
        M, _ = matrices
        # dot(v, M*w) — extract v
        expr = Dot(v, MVMul(M, w))
        result = extract_coeff(expr, v)
        # v is on the left, v == v → return right = MVMul(M, w)
        assert isinstance(result, MVMul)
        assert str(result.left) == 'M'
        assert str(result.right) == 'w'

    def test_cross_in_dot_extraction(self, vectors):
        """Extract vec from Dot(vec, Cross(a, b)) — common in angular momentum."""
        v, w, u = vectors
        expr = Dot(v, Cross(w, u))
        result = extract_coeff(expr, v)
        assert isinstance(result, Cross)
        assert str(result.left) == 'w'
        assert str(result.right) == 'u'

    def test_scalar_mul_dot_extraction(self, vectors):
        """Extract vec from s * Dot(vec, w)."""
        v, w, _ = vectors
        s = Scalar('s')
        expr = Mul(s, Dot(v, w))
        result = extract_coeff(expr, v)
        # extract_from_scalar(Mul(s, Dot(v, w)), v):
        #   right has v → SVMul(extract_from_scalar(Dot(v, w), v), s) = SVMul(w, s)
        assert isinstance(result, SVMul)
        assert str(result.left) == 'w'
        assert str(result.right) == 's'

    def test_add_of_dots_extraction(self, vectors):
        """Extract vec from Dot(vec, w) + Dot(vec, u)."""
        v, w, u = vectors
        expr = Add(Dot(v, w), Dot(v, u))
        result = extract_coeff(expr, v)
        assert isinstance(result, VAdd)
        assert len(result.nodes) == 2
        strs = {str(n) for n in result.nodes}
        assert strs == {'w', 'u'}

    def test_mvmul_extraction_from_vector(self, vectors, matrices):
        """Extract vec from MVMul(M, vec) directly."""
        v, _, _ = vectors
        M, _ = matrices
        expr = MVMul(M, v)
        result = extract_coeff(expr, v)
        assert str(result) == str(M)
