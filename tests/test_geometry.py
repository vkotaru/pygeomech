import pytest
from geomech.base.expr import Expression
from geomech.base.scalars import Scalar
from geomech.base.vectors import Vector, getVectors
from geomech.base.matrices import Matrix
from geomech.operations.addition import Add, VAdd
from geomech.operations.geometry import Delta, Dot, Cross, Hat, Vee
from geomech.operations.multiplication import Mul
from geomech.utils.errors import ExpressionMismatchError


class TestDelta:
    def test_creation(self):
        x = Vector('x')
        d = Delta(x)
        assert d.type == Expression.VECTOR
        assert str(d) == '\\delta{x}'

    def test_preserves_type_scalar(self):
        a = Scalar('a')
        d = Delta(a)
        assert d.type == Expression.SCALAR

    def test_preserves_type_matrix(self):
        M = Matrix('M')
        d = Delta(M)
        assert d.type == Expression.MATRIX

    def test_diff(self):
        x = Vector('x')
        d = Delta(x)
        result = d.diff()
        assert isinstance(result, Delta)

    def test_integrate(self):
        x = Vector('x')
        d = Delta(x)
        result = d.integrate()
        assert isinstance(result, Delta)


class TestDot:
    def test_creation(self):
        x, y = getVectors(['x', 'y'])
        d = Dot(x, y)
        assert d.type == Expression.SCALAR
        assert str(d) == 'Dot(x,y)'

    def test_is_constant_when_both_constant(self):
        e1 = Vector('e1', attr=['Constant'])
        e2 = Vector('e2', attr=['Constant'])
        d = Dot(e1, e2)
        assert d.isConstant

    def test_is_zero_when_one_zero(self):
        x = Vector('x')
        z = Vector('0', attr=['Constant', 'Zero'])
        d = Dot(x, z)
        assert d.isZero

    def test_rejects_non_vectors(self):
        a = Scalar('a')
        x = Vector('x')
        with pytest.raises(ExpressionMismatchError):
            Dot(a, x)

    def test_delta_both_variable(self):
        x, y = getVectors(['x', 'y'])
        d = Dot(x, y).delta()
        assert isinstance(d, Add)
        assert d.N == 2
        assert isinstance(d.nodes[0], Dot)
        assert isinstance(d.nodes[1], Dot)

    def test_delta_both_variable_str(self):
        x, y = getVectors(['x', 'y'])
        d = Dot(x, y).delta()
        assert str(d) == '(Dot(\\delta{x},y)+Dot(x,\\delta{y}))'

    def test_delta_constant_left(self):
        e3 = Vector('e3', attr=['Constant'])
        x = Vector('x')
        d = Dot(e3, x).delta()
        assert isinstance(d, Dot)

    def test_delta_constant_left_str(self):
        e3 = Vector('e3', attr=['Constant'])
        x = Vector('x')
        d = Dot(e3, x).delta()
        assert str(d) == 'Dot(e3,\\delta{x})'

    def test_delta_constant_right(self):
        x = Vector('x')
        e3 = Vector('e3', attr=['Constant'])
        d = Dot(x, e3).delta()
        assert isinstance(d, Dot)

    def test_delta_constant_right_str(self):
        x = Vector('x')
        e3 = Vector('e3', attr=['Constant'])
        d = Dot(x, e3).delta()
        assert str(d) == 'Dot(\\delta{x},e3)'

    def test_delta_self_dot(self):
        x = Vector('x')
        d = Dot(x, x).delta()
        # x.x variation = 2 * Dot(delta_x, x), returned as Mul(Dot(...), 2)
        assert isinstance(d, Mul)

    def test_delta_self_dot_str(self):
        x = Vector('x')
        d = Dot(x, x).delta()
        assert str(d) == 'Dot(\\delta{x},x)(2)'


class TestCross:
    def test_creation(self):
        x, y = getVectors(['x', 'y'])
        c = Cross(x, y)
        assert c.type == Expression.VECTOR
        assert str(c) == 'Cross(x,y)'

    def test_rejects_non_vectors(self):
        a = Scalar('a')
        x = Vector('x')
        with pytest.raises(ExpressionMismatchError):
            Cross(a, x)

    def test_delta_both_variable(self):
        x, y = getVectors(['x', 'y'])
        d = Cross(x, y).delta()
        assert isinstance(d, VAdd)
        assert d.N == 2
        assert isinstance(d.nodes[0], Cross)
        assert isinstance(d.nodes[1], Cross)

    def test_delta_both_variable_str(self):
        x, y = getVectors(['x', 'y'])
        d = Cross(x, y).delta()
        assert str(d) == '(Cross(\\delta{x},y)+Cross(x,\\delta{y}))'

    def test_delta_constant_left(self):
        e3 = Vector('e3', attr=['Constant'])
        x = Vector('x')
        d = Cross(e3, x).delta()
        assert isinstance(d, Cross)

    def test_delta_constant_left_str(self):
        e3 = Vector('e3', attr=['Constant'])
        x = Vector('x')
        d = Cross(e3, x).delta()
        assert str(d) == 'Cross(e3,\\delta{x})'

    def test_delta_constant_right(self):
        x = Vector('x')
        e3 = Vector('e3', attr=['Constant'])
        d = Cross(x, e3).delta()
        assert isinstance(d, Cross)

    def test_delta_constant_right_str(self):
        x = Vector('x')
        e3 = Vector('e3', attr=['Constant'])
        d = Cross(x, e3).delta()
        assert str(d) == 'Cross(\\delta{x},e3)'

    def test_diff_both_variable(self):
        x, y = getVectors(['x', 'y'])
        d = Cross(x, y).diff()
        assert isinstance(d, VAdd)
        assert d.N == 2
        assert isinstance(d.nodes[0], Cross)
        assert isinstance(d.nodes[1], Cross)

    def test_diff_both_variable_str(self):
        x, y = getVectors(['x', 'y'])
        d = Cross(x, y).diff()
        assert str(d) == '(Cross(dot_x,y)+Cross(x,dot_y))'

    def test_diff_constant_left(self):
        e3 = Vector('e3', attr=['Constant'])
        x = Vector('x')
        d = Cross(e3, x).diff()
        assert isinstance(d, Cross)

    def test_diff_constant_left_str(self):
        e3 = Vector('e3', attr=['Constant'])
        x = Vector('x')
        d = Cross(e3, x).diff()
        assert str(d) == 'Cross(e3,dot_x)'


class TestHat:
    def test_creation(self):
        x = Vector('x')
        h = Hat(x)
        assert h.type == Expression.MATRIX
        assert str(h) == 'Hat(x)'

    def test_rejects_non_vector(self):
        a = Scalar('a')
        with pytest.raises(ExpressionMismatchError):
            Hat(a)

    def test_delta(self):
        x = Vector('x')
        d = Hat(x).delta()
        assert isinstance(d, Hat)

    def test_delta_str(self):
        x = Vector('x')
        d = Hat(x).delta()
        assert str(d) == 'Hat(\\delta{x})'


class TestVee:
    def test_creation(self):
        M = Matrix('M')
        v = Vee(M)
        assert v.type == Expression.VECTOR
        assert str(v) == 'Vee(M)'

    def test_rejects_non_matrix(self):
        x = Vector('x')
        with pytest.raises(ExpressionMismatchError):
            Vee(x)
