import pytest

from geomech.core.base.expressions import (
    Matrix,
    Scalar,
    Vector,
    getMatrices,
    getScalars,
    getVectors,
)
from geomech.core.operations.addition import Add, MAdd, VAdd
from geomech.core.operations.calculus import TimeDerivative, TimeIntegral, Variation
from geomech.core.operations.geometry import Transpose
from geomech.core.operations.multiplication import (
    MMMul,
    Mul,
    MVMul,
    SMMul,
    SVMul,
    VVMul,
)
from geomech.utils.errors import ExpressionMismatchError

# ---------------------------------------------------------------------------
# __add__  (dispatched from _BaseMixin)
# ---------------------------------------------------------------------------


class TestMixinAdd:
    """__add__ on operation nodes routes through _BaseMixin."""

    # --- addition nodes (scalar / vector / matrix) ---
    def test_add_plus_scalar(self):
        a, b, c = getScalars("a b c")
        result = Add(a, b) + c
        assert isinstance(result, Add)

    def test_vadd_plus_vector(self):
        x, y, z = getVectors(["x", "y", "z"])
        result = VAdd(x, y) + z
        assert isinstance(result, VAdd)

    def test_madd_plus_matrix(self):
        A, B, C = getMatrices("A B C")
        result = MAdd(A, B) + C
        assert isinstance(result, MAdd)

    # --- multiplication nodes ---
    def test_mul_plus_scalar(self):
        a, b, c = getScalars("a b c")
        result = Mul(a, b) + c
        assert isinstance(result, Add)

    def test_svmul_plus_vector(self):
        x = Vector("x")
        a = Scalar("a")
        y = Vector("y")
        result = SVMul(x, a) + y
        assert isinstance(result, VAdd)

    def test_smmul_plus_matrix(self):
        M = Matrix("M")
        a = Scalar("a")
        N = Matrix("N")
        result = SMMul(M, a) + N
        assert isinstance(result, MAdd)

    # --- calculus nodes (scalar / vector / matrix inner) ---
    def test_variation_scalar_add(self):
        a, b = getScalars("a b")
        result = Variation(a) + b
        assert isinstance(result, Add)

    def test_variation_vector_add(self):
        x, y = getVectors(["x", "y"])
        result = Variation(x) + y
        assert isinstance(result, VAdd)

    def test_variation_matrix_add(self):
        M, N = getMatrices("M N")
        result = Variation(M) + N
        assert isinstance(result, MAdd)

    def test_timederivative_add(self):
        a, b = getScalars("a b")
        result = TimeDerivative(a) + b
        assert isinstance(result, Add)

    def test_timeintegral_add(self):
        x, y = getVectors(["x", "y"])
        result = TimeIntegral(x) + y
        assert isinstance(result, VAdd)

    # --- VVMul (dynamic result type) ---
    def test_vvmul_inner_plus_scalar(self):
        x, y = getVectors(["x", "y"])
        a = Scalar("a")
        inner = VVMul(Transpose(x), y)
        result = inner + a
        assert isinstance(result, Add)

    def test_vvmul_outer_plus_matrix(self):
        x, y = getVectors(["x", "y"])
        M = Matrix("M")
        outer = VVMul(x, Transpose(y))
        result = outer + M
        assert isinstance(result, MAdd)


# ---------------------------------------------------------------------------
# __iadd__
# ---------------------------------------------------------------------------


class TestMixinIadd:
    def test_iadd_scalar(self):
        a, b = getScalars("a b")
        result = Mul(a, b)
        result += a
        assert isinstance(result, Add)

    def test_iadd_vector(self):
        x, y = getVectors(["x", "y"])
        result = Variation(x)
        result += y
        assert isinstance(result, VAdd)


# ---------------------------------------------------------------------------
# __sub__  (new — was missing on calculus nodes and VVMul)
# ---------------------------------------------------------------------------


class TestMixinSub:
    """__sub__ on operation nodes routes through _BaseMixin."""

    # --- addition nodes ---
    def test_add_sub_scalar(self):
        a, b, c = getScalars("a b c")
        result = Add(a, b) - c
        assert isinstance(result, Add)
        assert isinstance(result.nodes[-1], Mul)  # c * (-1)

    def test_vadd_sub_vector(self):
        x, y, z = getVectors(["x", "y", "z"])
        result = VAdd(x, y) - z
        assert isinstance(result, VAdd)
        assert isinstance(result.nodes[-1], SVMul)  # z * (-1)

    def test_madd_sub_matrix(self):
        A, B, C = getMatrices("A B C")
        result = MAdd(A, B) - C
        assert isinstance(result, MAdd)
        assert isinstance(result.nodes[-1], SMMul)  # C * (-1)

    # --- multiplication nodes ---
    def test_mul_sub_scalar(self):
        a, b, c = getScalars("a b c")
        result = Mul(a, b) - c
        assert isinstance(result, Add)

    def test_svmul_sub_vector(self):
        x = Vector("x")
        a = Scalar("a")
        y = Vector("y")
        result = SVMul(x, a) - y
        assert isinstance(result, VAdd)

    # --- calculus nodes ---
    def test_variation_scalar_sub(self):
        a, b = getScalars("a b")
        result = Variation(a) - b
        assert isinstance(result, Add)
        assert isinstance(result.nodes[1], Mul)

    def test_variation_vector_sub(self):
        x, y = getVectors(["x", "y"])
        result = Variation(x) - y
        assert isinstance(result, VAdd)
        assert isinstance(result.nodes[1], SVMul)

    def test_variation_matrix_sub(self):
        M, N = getMatrices("M N")
        result = Variation(M) - N
        assert isinstance(result, MAdd)
        assert isinstance(result.nodes[1], SMMul)

    def test_timederivative_sub(self):
        a, b = getScalars("a b")
        result = TimeDerivative(a) - b
        assert isinstance(result, Add)

    def test_timeintegral_sub(self):
        x, y = getVectors(["x", "y"])
        result = TimeIntegral(x) - y
        assert isinstance(result, VAdd)

    # --- VVMul ---
    def test_vvmul_inner_sub_scalar(self):
        x, y = getVectors(["x", "y"])
        a = Scalar("a")
        inner = VVMul(Transpose(x), y)
        result = inner - a
        assert isinstance(result, Add)

    def test_vvmul_outer_sub_matrix(self):
        x, y = getVectors(["x", "y"])
        M = Matrix("M")
        outer = VVMul(x, Transpose(y))
        result = outer - M
        assert isinstance(result, MAdd)

    # --- type mismatch ---
    def test_sub_type_mismatch_scalar_vector(self):
        a = Scalar("a")
        x = Vector("x")
        with pytest.raises(ExpressionMismatchError):
            Mul(a, a) - x

    def test_sub_type_mismatch_vector_matrix(self):
        x = Vector("x")
        M = Matrix("M")
        with pytest.raises(ExpressionMismatchError):
            Variation(x) - M

    def test_sub_type_mismatch_matrix_scalar(self):
        M = Matrix("M")
        a = Scalar("a")
        with pytest.raises(ExpressionMismatchError):
            MAdd(M, M) - a


# ---------------------------------------------------------------------------
# __mul__  (dispatched from _BaseMixin)
# ---------------------------------------------------------------------------


class TestMixinMul:
    """__mul__ on operation nodes routes through _BaseMixin."""

    # --- scalar * scalar ---
    def test_add_mul_scalar(self):
        a, b, c = getScalars("a b c")
        result = Add(a, b) * c
        assert isinstance(result, Mul)

    # --- scalar * vector ---
    def test_add_mul_vector(self):
        a, b = getScalars("a b")
        x = Vector("x")
        result = Add(a, b) * x
        assert isinstance(result, SVMul)

    # --- scalar * matrix ---
    def test_add_mul_matrix(self):
        a, b = getScalars("a b")
        M = Matrix("M")
        result = Add(a, b) * M
        assert isinstance(result, SMMul)

    # --- vector * scalar ---
    def test_vadd_mul_scalar(self):
        x, y = getVectors(["x", "y"])
        a = Scalar("a")
        result = VAdd(x, y) * a
        assert isinstance(result, SVMul)

    # --- matrix * scalar ---
    def test_madd_mul_scalar(self):
        A, B = getMatrices("A B")
        a = Scalar("a")
        result = MAdd(A, B) * a
        assert isinstance(result, SMMul)

    # --- matrix * vector ---
    def test_madd_mul_vector(self):
        A, B = getMatrices("A B")
        x = Vector("x")
        result = MAdd(A, B) * x
        assert isinstance(result, MVMul)

    # --- matrix * matrix ---
    def test_madd_mul_matrix(self):
        A, B, C = getMatrices("A B C")
        result = MAdd(A, B) * C
        assert isinstance(result, MMMul)

    # --- calculus nodes ---
    def test_variation_scalar_mul_scalar(self):
        a, b = getScalars("a b")
        result = Variation(a) * b
        assert isinstance(result, Mul)

    def test_variation_scalar_mul_vector(self):
        a = Scalar("a")
        x = Vector("x")
        result = Variation(a) * x
        assert isinstance(result, SVMul)

    def test_variation_vector_mul_scalar(self):
        x = Vector("x")
        a = Scalar("a")
        result = Variation(x) * a
        assert isinstance(result, SVMul)

    def test_variation_matrix_mul_vector(self):
        M = Matrix("M")
        x = Vector("x")
        result = Variation(M) * x
        assert isinstance(result, MVMul)

    def test_timederivative_mul_numeric(self):
        a = Scalar("a")
        result = TimeDerivative(a) * 3
        assert isinstance(result, Mul)

    # --- VVMul (dynamic type) ---
    def test_vvmul_inner_mul_scalar(self):
        x, y = getVectors(["x", "y"])
        a = Scalar("a")
        inner = VVMul(Transpose(x), y)
        result = inner * a
        assert isinstance(result, Mul)

    def test_vvmul_inner_mul_vector(self):
        x, y = getVectors(["x", "y"])
        z = Vector("z")
        inner = VVMul(Transpose(x), y)
        result = inner * z
        assert isinstance(result, SVMul)

    def test_vvmul_outer_mul_scalar(self):
        x, y = getVectors(["x", "y"])
        a = Scalar("a")
        outer = VVMul(x, Transpose(y))
        result = outer * a
        assert isinstance(result, SMMul)

    def test_vvmul_outer_mul_vector(self):
        x, y = getVectors(["x", "y"])
        z = Vector("z")
        outer = VVMul(x, Transpose(y))
        result = outer * z
        assert isinstance(result, MVMul)

    def test_vvmul_outer_mul_matrix(self):
        x, y = getVectors(["x", "y"])
        M = Matrix("M")
        outer = VVMul(x, Transpose(y))
        result = outer * M
        assert isinstance(result, MMMul)

    # --- numeric coercion ---
    def test_mul_numeric_int(self):
        a, b = getScalars("a b")
        result = Add(a, b) * 5
        assert isinstance(result, Mul)

    def test_mul_numeric_float(self):
        x, y = getVectors(["x", "y"])
        result = VAdd(x, y) * 2.5
        assert isinstance(result, SVMul)
