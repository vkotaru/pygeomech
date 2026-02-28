import pytest
from geomech.core.base.expressions import (
    Scalar, Vector, Matrix, S2, SO3, TS2, TSO3,
)
from geomech.core.operations.calculus import Variation


# ---------------------------------------------------------------------------
# get_variation_vector
# ---------------------------------------------------------------------------

class TestGetVariationVector:
    def test_so3_returns_eta(self):
        """SO3('R').get_variation_vector() → Vector('\\eta_{R}')."""
        R = SO3('R')
        eta = R.get_variation_vector()
        assert isinstance(eta, Vector)
        assert eta.name == '\\eta_{R}'

    def test_s2_returns_xi(self):
        """S2('q').get_variation_vector() → Vector('\\xi_{q}')."""
        q = S2('q')
        xi = q.get_variation_vector()
        assert isinstance(xi, Vector)
        assert xi.name == '\\xi_{q}'

    def test_vector_returns_delta(self):
        """Vector('x').get_variation_vector() → Variation(x)."""
        x = Vector('x')
        result = x.get_variation_vector()
        assert isinstance(result, Variation)

    def test_constant_vector_returns_zero(self):
        """Constant vector variation is zero."""
        c = Vector('c', attr=['Constant'])
        result = c.get_variation_vector()
        assert result.is_zero

    def test_scalar_returns_delta(self):
        """Scalar('a').get_variation_vector() → Variation(a) via Expr default."""
        a = Scalar('a')
        result = a.get_variation_vector()
        assert isinstance(result, Variation)

    def test_matrix_returns_delta(self):
        """Matrix('M').get_variation_vector() → Variation(M) via Expr default."""
        M = Matrix('M')
        result = M.get_variation_vector()
        assert isinstance(result, Variation)


# ---------------------------------------------------------------------------
# get_tangent_vector
# ---------------------------------------------------------------------------

class TestGetTangentVector:
    def test_so3_returns_tso3(self):
        """SO3('R').get_tangent_vector() → TSO3('\\Omega_{R}')."""
        R = SO3('R')
        omega = R.get_tangent_vector()
        assert isinstance(omega, TSO3)
        assert omega.name == '\\Omega_{R}'
        assert omega.SO3 is R

    def test_s2_returns_ts2(self):
        """S2('q').get_tangent_vector() → TS2('\\omega_{q}')."""
        q = S2('q')
        omega = q.get_tangent_vector()
        assert isinstance(omega, TS2)
        assert omega.name == '\\omega_{q}'
        assert omega.S2 is q

    def test_plain_vector_raises(self):
        """Plain Vector has no tangent vector — raises NotImplementedError."""
        x = Vector('x')
        with pytest.raises(NotImplementedError):
            x.get_tangent_vector()

    def test_scalar_raises(self):
        """Scalar has no tangent vector — raises NotImplementedError."""
        a = Scalar('a')
        with pytest.raises(NotImplementedError):
            a.get_tangent_vector()

    def test_matrix_raises(self):
        """Plain Matrix has no tangent vector — raises NotImplementedError."""
        M = Matrix('M')
        with pytest.raises(NotImplementedError):
            M.get_tangent_vector()
