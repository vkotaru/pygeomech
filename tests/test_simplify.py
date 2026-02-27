"""Tests for simplify() and full_simplify()."""

import pytest
from geomech.core.base.expressions import (
    Scalar, Vector, Matrix,
    Zero, ZeroVector, ZeroMatrix,
    getScalars, getVectors, getMatrices,
)
from geomech.core.operations.addition import Add, VAdd, MAdd
from geomech.core.operations.multiplication import (
    Mul, SVMul, SMMul, MVMul, MMMul,
)
from geomech.core.operations.geometry import Dot, Cross, Hat, Vee, Transpose
from geomech.core.operations.calculus import Variation, TimeDerivative, TimeIntegral
from geomech.core.transformations.simplify import simplify, full_simplify


# ===================================================================
# Scalar Add
# ===================================================================

class TestSimplifyScalarAdd:
    def test_remove_zeros(self):
        """Add(a, 0) simplifies to a."""
        a = Scalar('a')
        result = simplify(Add(a, Zero))
        assert result == a

    def test_all_zeros(self):
        """Add(0, 0) → Zero."""
        result = simplify(Add(Zero, Zero))
        assert result == Zero

    def test_combine_numerics(self):
        """Add(Scalar(2), Scalar(3)) → Scalar(5)."""
        two = Scalar('(2)', value=2, attr=['Constant'])
        three = Scalar('(3)', value=3, attr=['Constant'])
        result = simplify(Add(two, three))
        assert result.value == 5

    def test_combine_numerics_with_symbolic(self):
        """Add(a, 2, 3) → Add(a, 5)."""
        a = Scalar('a')
        two = Scalar('(2)', value=2, attr=['Constant'])
        three = Scalar('(3)', value=3, attr=['Constant'])
        result = simplify(Add(a, two, three))
        assert isinstance(result, Add)
        assert len(result.nodes) == 2

    def test_single_node_unwrap(self):
        """Add(a) → a."""
        a = Scalar('a')
        result = simplify(Add(a, Zero))
        assert result == a
        assert not isinstance(result, Add)

    def test_zero_numerics_not_added(self):
        """Add(a, 0_numeric) keeps only a."""
        a = Scalar('a')
        zero_val = Scalar('0', value=0, attr=['Constant', 'Zero'])
        result = simplify(Add(a, zero_val))
        assert result == a


# ===================================================================
# Vector Add
# ===================================================================

class TestSimplifyVectorAdd:
    def test_remove_zeros(self):
        """VAdd(x, 0v) simplifies to x."""
        x = Vector('x')
        result = simplify(VAdd(x, ZeroVector))
        assert result == x

    def test_all_zeros(self):
        """VAdd(0v, 0v) → ZeroVector."""
        result = simplify(VAdd(ZeroVector, ZeroVector))
        assert _is_zero(result)

    def test_single_node_unwrap(self):
        """VAdd(x) after filtering → x."""
        x = Vector('x')
        result = simplify(VAdd(x, ZeroVector))
        assert not isinstance(result, VAdd)
        assert result == x


# ===================================================================
# Matrix Add
# ===================================================================

class TestSimplifyMatrixAdd:
    def test_remove_zeros(self):
        """MAdd(M, 0m) simplifies to M."""
        M = Matrix('M')
        result = simplify(MAdd(M, ZeroMatrix))
        assert result == M

    def test_all_zeros(self):
        """MAdd(0m, 0m) → ZeroMatrix."""
        result = simplify(MAdd(ZeroMatrix, ZeroMatrix))
        assert _is_zero(result)

    def test_single_node_unwrap(self):
        """MAdd(M) after filtering → M."""
        M = Matrix('M')
        result = simplify(MAdd(M, ZeroMatrix))
        assert not isinstance(result, MAdd)
        assert result == M


# ===================================================================
# Scalar Mul
# ===================================================================

class TestSimplifyScalarMul:
    def test_zero_left(self):
        """Mul(0, a) → Zero."""
        a = Scalar('a')
        result = simplify(Mul(Zero, a))
        assert _is_zero(result)

    def test_zero_right(self):
        """Mul(a, 0) → Zero."""
        a = Scalar('a')
        result = simplify(Mul(a, Zero))
        assert _is_zero(result)

    def test_identity_left(self):
        """Mul(1, a) → a."""
        a = Scalar('a')
        one = Scalar('1', value=1, attr=['Constant', 'Ones'])
        result = simplify(Mul(one, a))
        assert result == a

    def test_identity_right(self):
        """Mul(a, 1) → a."""
        a = Scalar('a')
        one = Scalar('1', value=1, attr=['Constant', 'Ones'])
        result = simplify(Mul(a, one))
        assert result == a

    def test_combine_numerics(self):
        """Mul(Scalar(2), Scalar(3)) → Scalar(6)."""
        two = Scalar('(2)', value=2, attr=['Constant'])
        three = Scalar('(3)', value=3, attr=['Constant'])
        result = simplify(Mul(two, three))
        assert result.value == 6

    def test_symbolic_unchanged(self):
        """Mul(a, b) stays as Mul when both symbolic."""
        a, b = getScalars('a b')
        result = simplify(Mul(a, b))
        assert isinstance(result, Mul)


# ===================================================================
# SVMul
# ===================================================================

class TestSimplifySVMul:
    def test_zero_scalar(self):
        """SVMul(x, 0) → ZeroVector."""
        x = Vector('x')
        result = simplify(SVMul(x, Zero))
        assert _is_zero(result)

    def test_zero_vector(self):
        """SVMul(0v, a) → ZeroVector."""
        a = Scalar('a')
        result = simplify(SVMul(ZeroVector, a))
        assert _is_zero(result)

    def test_identity(self):
        """SVMul(x, 1) → x."""
        x = Vector('x')
        one = Scalar('1', value=1, attr=['Constant', 'Ones'])
        result = simplify(SVMul(x, one))
        assert result == x

    def test_symbolic_unchanged(self):
        """SVMul(x, a) stays when both non-trivial."""
        x = Vector('x')
        a = Scalar('a')
        result = simplify(SVMul(x, a))
        assert isinstance(result, SVMul)


# ===================================================================
# SMMul
# ===================================================================

class TestSimplifySMMul:
    def test_zero_scalar(self):
        """SMMul(M, 0) → ZeroMatrix."""
        M = Matrix('M')
        result = simplify(SMMul(M, Zero))
        assert _is_zero(result)

    def test_zero_matrix(self):
        """SMMul(0m, a) → ZeroMatrix."""
        a = Scalar('a')
        result = simplify(SMMul(ZeroMatrix, a))
        assert _is_zero(result)

    def test_identity(self):
        """SMMul(M, 1) → M."""
        M = Matrix('M')
        one = Scalar('1', value=1, attr=['Constant', 'Ones'])
        result = simplify(SMMul(M, one))
        assert result == M

    def test_symbolic_unchanged(self):
        """SMMul(M, a) stays when both non-trivial."""
        M = Matrix('M')
        a = Scalar('a')
        result = simplify(SMMul(M, a))
        assert isinstance(result, SMMul)


# ===================================================================
# MVMul
# ===================================================================

class TestSimplifyMVMul:
    def test_zero_matrix(self):
        """MVMul(0m, x) → ZeroVector."""
        x = Vector('x')
        result = simplify(MVMul(ZeroMatrix, x))
        assert _is_zero(result)

    def test_zero_vector(self):
        """MVMul(M, 0v) → ZeroVector."""
        M = Matrix('M')
        result = simplify(MVMul(M, ZeroVector))
        assert _is_zero(result)

    def test_identity_matrix(self):
        """MVMul(I, x) → x."""
        from geomech.core.base.expressions import IdentityMatrix
        x = Vector('x')
        result = simplify(MVMul(IdentityMatrix, x))
        assert result == x

    def test_symbolic_unchanged(self):
        """MVMul(M, x) stays when both non-trivial."""
        M = Matrix('M')
        x = Vector('x')
        result = simplify(MVMul(M, x))
        assert isinstance(result, MVMul)


# ===================================================================
# MMMul
# ===================================================================

class TestSimplifyMMMul:
    def test_zero_left(self):
        """MMMul(0m, M) → ZeroMatrix."""
        M = Matrix('M')
        result = simplify(MMMul(ZeroMatrix, M))
        assert _is_zero(result)

    def test_zero_right(self):
        """MMMul(M, 0m) → ZeroMatrix."""
        M = Matrix('M')
        result = simplify(MMMul(M, ZeroMatrix))
        assert _is_zero(result)

    def test_identity_left(self):
        """MMMul(I, M) → M."""
        from geomech.core.base.expressions import IdentityMatrix
        M = Matrix('M')
        result = simplify(MMMul(IdentityMatrix, M))
        assert result == M

    def test_identity_right(self):
        """MMMul(M, I) → M."""
        from geomech.core.base.expressions import IdentityMatrix
        M = Matrix('M')
        result = simplify(MMMul(M, IdentityMatrix))
        assert result == M


# ===================================================================
# Dot
# ===================================================================

class TestSimplifyDot:
    def test_zero_left(self):
        """Dot(0v, x) → Zero."""
        x = Vector('x')
        result = simplify(Dot(ZeroVector, x))
        assert _is_zero(result)

    def test_zero_right(self):
        """Dot(x, 0v) → Zero."""
        x = Vector('x')
        result = simplify(Dot(x, ZeroVector))
        assert _is_zero(result)

    def test_symbolic_unchanged(self):
        """Dot(x, y) stays when both non-trivial."""
        x, y = getVectors(['x', 'y'])
        result = simplify(Dot(x, y))
        assert isinstance(result, Dot)


# ===================================================================
# Cross
# ===================================================================

class TestSimplifyCross:
    def test_zero_left(self):
        """Cross(0v, x) → ZeroVector."""
        x = Vector('x')
        result = simplify(Cross(ZeroVector, x))
        assert _is_zero(result)

    def test_zero_right(self):
        """Cross(x, 0v) → ZeroVector."""
        x = Vector('x')
        result = simplify(Cross(x, ZeroVector))
        assert _is_zero(result)


# ===================================================================
# Unary geometry ops
# ===================================================================

class TestSimplifyUnary:
    def test_hat_zero(self):
        """Hat(0v) → ZeroMatrix."""
        result = simplify(Hat(ZeroVector))
        assert _is_zero(result)

    def test_hat_nonzero(self):
        """Hat(x) stays when x is not zero."""
        x = Vector('x')
        result = simplify(Hat(x))
        assert isinstance(result, Hat)

    def test_vee_zero(self):
        """Vee(0m) → ZeroVector."""
        result = simplify(Vee(ZeroMatrix))
        assert _is_zero(result)

    def test_vee_nonzero(self):
        """Vee(M) stays when M is not zero."""
        M = Matrix('M')
        result = simplify(Vee(M))
        assert isinstance(result, Vee)

    def test_hat_vee_cancel(self):
        """Hat(Vee(M)) → M."""
        M = Matrix('M')
        result = simplify(Hat(Vee(M)))
        assert result == M

    def test_vee_hat_cancel(self):
        """Vee(Hat(x)) → x."""
        x = Vector('x')
        result = simplify(Vee(Hat(x)))
        assert result == x

    def test_transpose_passthrough(self):
        """Transpose(x) recurses but preserves structure."""
        x = Vector('x')
        result = simplify(Transpose(x))
        assert isinstance(result, Transpose)

    def test_double_transpose_cancel(self):
        """Transpose(Transpose(x)) → x."""
        x = Vector('x')
        result = simplify(Transpose(Transpose(x)))
        assert result == x


# ===================================================================
# Calculus ops
# ===================================================================

class TestSimplifyCalculus:
    def test_variation_zero(self):
        """Variation(0v) → ZeroVector."""
        result = simplify(Variation(ZeroVector))
        assert _is_zero(result)

    def test_variation_nonzero(self):
        """Variation(x) stays when x is not zero."""
        x = Vector('x')
        result = simplify(Variation(x))
        assert isinstance(result, Variation)

    def test_time_derivative_zero(self):
        """TimeDerivative(0) → Zero."""
        result = simplify(TimeDerivative(Zero))
        assert _is_zero(result)

    def test_time_integral_zero(self):
        """TimeIntegral(0) → Zero."""
        result = simplify(TimeIntegral(Zero))
        assert _is_zero(result)


# ===================================================================
# Leaf passthrough
# ===================================================================

class TestSimplifyLeaves:
    def test_scalar(self):
        """Scalar leaf passes through unchanged."""
        a = Scalar('a')
        assert simplify(a) is a

    def test_vector(self):
        """Vector leaf passes through unchanged."""
        x = Vector('x')
        assert simplify(x) is x

    def test_matrix(self):
        """Matrix leaf passes through unchanged."""
        M = Matrix('M')
        assert simplify(M) is M


# ===================================================================
# Recursive / nested simplification
# ===================================================================

class TestSimplifyRecursive:
    def test_nested_add_with_zeros(self):
        """Add(Add(a, 0), 0) → a  (bottom-up zero removal)."""
        a = Scalar('a')
        inner = Add(a, Zero)
        result = simplify(Add(inner, Zero))
        assert result == a

    def test_mul_nested_zero(self):
        """Mul(a, Mul(b, 0)) → Zero  (inner zero propagates)."""
        a, b = getScalars('a b')
        result = simplify(Mul(a, Mul(b, Zero)))
        assert _is_zero(result)

    def test_svmul_nested_zero_scalar(self):
        """SVMul(x, Mul(a, 0)) → ZeroVector  (inner scalar simplifies to zero)."""
        x = Vector('x')
        a = Scalar('a')
        result = simplify(SVMul(x, Mul(a, Zero)))
        assert _is_zero(result)

    def test_dot_nested_zero(self):
        """Dot(SVMul(x, 0), y) → Zero  (inner SVMul simplifies to zero)."""
        x, y = getVectors(['x', 'y'])
        result = simplify(Dot(SVMul(x, Zero), y))
        assert _is_zero(result)

    def test_chain_mul_numerics(self):
        """Mul(Mul(2, 3), a) → Mul(6, a)  (bottom-up combines inner first)."""
        two = Scalar('(2)', value=2, attr=['Constant'])
        three = Scalar('(3)', value=3, attr=['Constant'])
        a = Scalar('a')
        result = simplify(Mul(Mul(two, three), a))
        assert isinstance(result, Mul)
        assert result.left.value == 6

    def test_add_with_nested_mul_zero(self):
        """Add(a, Mul(b, 0)) → a  (inner Mul becomes Zero, then filtered)."""
        a, b = getScalars('a b')
        result = simplify(Add(a, Mul(b, Zero)))
        assert result == a


# ===================================================================
# full_simplify
# ===================================================================

class TestFullSimplify:
    def test_basic_expansion_and_zero(self):
        """full_simplify expands then eliminates zeros."""
        a = Scalar('a')
        x, y = getVectors(['x', 'y'])
        # a * (x + 0v) → expand → Add(SVMul(x,a), SVMul(0v,a))
        #                → simplify → SVMul(x, a)
        expr = SVMul(VAdd(x, ZeroVector), a)
        result = full_simplify(expr)
        assert isinstance(result, SVMul)
        assert result.left == x

    def test_convergence(self):
        """full_simplify converges (applying simplify twice gives same result)."""
        a = Scalar('a')
        x = Vector('x')
        expr = SVMul(x, a)
        result = full_simplify(expr)
        assert result == full_simplify(result)

    def test_max_iter_respected(self):
        """full_simplify with max_iter=1 does at most one simplify pass."""
        a = Scalar('a')
        result = full_simplify(a, max_iter=1)
        assert result == a

    def test_pull_then_simplify(self):
        """full_simplify pulls scalars then simplifies.

        Dot(SVMul(x, a), y) → pull → Mul(a, Dot(x, y))
        """
        a = Scalar('a')
        x, y = getVectors(['x', 'y'])
        expr = Dot(SVMul(x, a), y)
        result = full_simplify(expr)
        assert isinstance(result, Mul)

    def test_vector_rules_then_simplify(self):
        """full_simplify applies vector_rules then simplifies.

        Add(Dot(x, Cross(x, y)), a) → vrules makes dot=0 → simplify filters → a
        """
        a = Scalar('a')
        x, y = getVectors(['x', 'y'])
        expr = Add(Dot(x, Cross(x, y)), a)
        result = full_simplify(expr)
        assert result == a


# ===================================================================
# Helpers (used in tests)
# ===================================================================

def _is_zero(expr) -> bool:
    return getattr(expr, 'isZero', False)
