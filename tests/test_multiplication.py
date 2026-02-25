import pytest
from geomech.base.expr import Expression
from geomech.base.scalars import Scalar, getScalars
from geomech.base.vectors import Vector, getVectors
from geomech.base.matrices import Matrix, getMatrices
from geomech.operations.addition import Add, VAdd, MAdd
from geomech.operations.multiplication import Mul, SVMul, SMMul, MVMul, MMMul, VVMul
from geomech.operations.transpose import Transpose


class TestMul:
    def test_creation(self):
        a, b = getScalars('a b')
        result = Mul(a, b)
        assert result.type == Expression.SCALAR
        assert str(result) == 'ab'

    def test_int_coercion(self):
        a = Scalar('a')
        result = Mul(a, 2)
        assert isinstance(result, Mul)
        assert str(result) == 'a(2)'

    def test_float_coercion(self):
        a = Scalar('a')
        result = Mul(a, 0.5)
        assert isinstance(result, Mul)
        assert str(result) == 'a(0.5)'

    def test_constant_propagation(self):
        m = Scalar('m', attr=['Constant'])
        g = Scalar('g', attr=['Constant'])
        result = Mul(m, g)
        assert result.isConstant
        assert str(result) == 'mg'

    def test_zero_propagation(self):
        a = Scalar('a')
        z = Scalar('0', attr=['Constant', 'Zero'])
        result = Mul(a, z)
        assert result.isZero
        assert str(result) == 'a0'


class TestMulProductRule:
    def test_delta_both_variable(self):
        a, b = getScalars('a b')
        d = (a * b).delta()
        assert isinstance(d, Add)
        assert d.N == 2
        assert isinstance(d.nodes[0], Mul)
        assert isinstance(d.nodes[1], Mul)
        assert d.nodes[0].left == a.delta()
        assert d.nodes[0].right == b
        assert d.nodes[1].left == a
        assert d.nodes[1].right == b.delta()

    def test_delta_both_variable_str(self):
        a, b = getScalars('a b')
        d = (a * b).delta()
        assert str(d) == '(\\delta{a}b+a\\delta{b})'

    def test_delta_constant_left(self):
        m = Scalar('m', attr=['Constant'])
        a = Scalar('a')
        d = (m * a).delta()
        assert isinstance(d, Mul)
        assert d.left == m
        assert d.right == a.delta()

    def test_delta_constant_left_str(self):
        m = Scalar('m', attr=['Constant'])
        a = Scalar('a')
        d = (m * a).delta()
        assert str(d) == 'm\\delta{a}'

    def test_delta_constant_right(self):
        a = Scalar('a')
        m = Scalar('m', attr=['Constant'])
        d = (a * m).delta()
        assert isinstance(d, Mul)
        assert d.left == a.delta()
        assert d.right == m

    def test_delta_constant_right_str(self):
        a = Scalar('a')
        m = Scalar('m', attr=['Constant'])
        d = (a * m).delta()
        assert str(d) == '\\delta{a}m'

    def test_delta_both_constant(self):
        m = Scalar('m', attr=['Constant'])
        g = Scalar('g', attr=['Constant'])
        d = (m * g).delta()
        assert d.isZero

    def test_diff_both_variable(self):
        a, b = getScalars('a b')
        d = (a * b).diff()
        assert isinstance(d, Add)
        assert d.N == 2
        assert isinstance(d.nodes[0], Mul)
        assert isinstance(d.nodes[1], Mul)
        assert d.nodes[0].left == a.diff()
        assert d.nodes[0].right == b
        assert d.nodes[1].left == a
        assert d.nodes[1].right == b.diff()

    def test_diff_both_variable_str(self):
        a, b = getScalars('a b')
        d = (a * b).diff()
        assert str(d) == '(dot_ab+adot_b)'

    def test_diff_constant_left(self):
        m = Scalar('m', attr=['Constant'])
        a = Scalar('a')
        d = (m * a).diff()
        assert isinstance(d, Mul)
        assert d.left == m
        assert d.right == a.diff()

    def test_diff_constant_left_str(self):
        m = Scalar('m', attr=['Constant'])
        a = Scalar('a')
        d = (m * a).diff()
        assert str(d) == 'mdot_a'

    def test_diff_both_constant(self):
        m = Scalar('m', attr=['Constant'])
        g = Scalar('g', attr=['Constant'])
        d = (m * g).diff()
        assert d.isZero


class TestSVMul:
    def test_creation(self):
        x = Vector('x')
        a = Scalar('a')
        result = SVMul(x, a)
        assert result.type == Expression.VECTOR
        assert str(result) == 'xa'
        assert result.left == x
        assert result.right == a

    def test_normalizes_order(self):
        """SVMul always stores vector left, scalar right."""
        x = Vector('x')
        a = Scalar('a')
        result = SVMul(a, x)
        assert result.left == x
        assert result.right == a
        assert result.type == Expression.VECTOR
        assert str(result) == 'xa'

    def test_delta_product_rule(self):
        a = Scalar('a')
        x = Vector('x')
        d = (a * x).delta()
        assert isinstance(d, VAdd)
        assert d.N == 2
        assert isinstance(d.nodes[0], SVMul)
        assert isinstance(d.nodes[1], SVMul)
        assert d.nodes[0].left == x.delta()
        assert d.nodes[0].right == a
        assert d.nodes[1].left == x
        assert d.nodes[1].right == a.delta()

    def test_delta_product_rule_str(self):
        a = Scalar('a')
        x = Vector('x')
        d = (a * x).delta()
        assert str(d) == '(\\delta{x}a+x\\delta{a})'

    def test_delta_constant_scalar(self):
        m = Scalar('m', attr=['Constant'])
        x = Vector('x')
        d = (m * x).delta()
        assert isinstance(d, SVMul)
        assert d.left == x.delta()
        assert d.right == m

    def test_delta_constant_scalar_str(self):
        m = Scalar('m', attr=['Constant'])
        x = Vector('x')
        d = (m * x).delta()
        assert str(d) == '\\delta{x}m'

    def test_delta_both_constant(self):
        m = Scalar('m', attr=['Constant'])
        x = Vector('x', attr=['Constant'])
        d = (m * x).delta()
        assert d.isZero


class TestSMMul:
    def test_creation(self):
        M = Matrix('M')
        a = Scalar('a')
        result = SMMul(M, a)
        assert result.type == Expression.MATRIX
        assert str(result) == 'Ma'
        assert result.left == M
        assert result.right == a

    def test_normalizes_order(self):
        """SMMul always stores matrix left, scalar right."""
        M = Matrix('M')
        a = Scalar('a')
        result = SMMul(a, M)
        assert result.left == M
        assert result.right == a
        assert result.type == Expression.MATRIX
        assert str(result) == 'Ma'

    def test_delta_product_rule(self):
        a = Scalar('a')
        M = Matrix('M')
        d = (a * M).delta()
        assert isinstance(d, MAdd)
        assert d.N == 2
        assert isinstance(d.nodes[0], SMMul)
        assert isinstance(d.nodes[1], SMMul)
        assert d.nodes[0].left == M.delta()
        assert d.nodes[0].right == a
        assert d.nodes[1].left == M
        assert d.nodes[1].right == a.delta()

    def test_delta_product_rule_str(self):
        a = Scalar('a')
        M = Matrix('M')
        d = (a * M).delta()
        assert str(d) == '(\\delta{M}a+M\\delta{a})'

    def test_delta_constant_scalar(self):
        m = Scalar('m', attr=['Constant'])
        M = Matrix('M')
        d = (m * M).delta()
        assert isinstance(d, SMMul)
        assert d.left == M.delta()
        assert d.right == m

    def test_delta_constant_scalar_str(self):
        m = Scalar('m', attr=['Constant'])
        M = Matrix('M')
        d = (m * M).delta()
        assert str(d) == '\\delta{M}m'

    def test_delta_both_constant(self):
        m = Scalar('m', attr=['Constant'])
        M = Matrix('M', attr=['Constant'])
        d = (m * M).delta()
        assert d.isZero


class TestMVMul:
    def test_creation(self):
        M = Matrix('M')
        x = Vector('x')
        result = MVMul(M, x)
        assert result.type == Expression.VECTOR

    def test_delta_product_rule(self):
        M = Matrix('M')
        x = Vector('x')
        d = (M * x).delta()
        assert isinstance(d, VAdd)
        assert d.N == 2
        assert isinstance(d.nodes[0], MVMul)
        assert isinstance(d.nodes[1], MVMul)
        assert d.nodes[0].left == M.delta()
        assert d.nodes[0].right == x
        assert d.nodes[1].left == M
        assert d.nodes[1].right == x.delta()

    def test_delta_product_rule_str(self):
        M = Matrix('M')
        x = Vector('x')
        d = (M * x).delta()
        assert str(d) == '(\\delta{M}x+M\\delta{x})'

    def test_delta_constant_matrix(self):
        J = Matrix('J', attr=['Constant'])
        x = Vector('x')
        d = (J * x).delta()
        assert isinstance(d, MVMul)
        assert d.left == J
        assert d.right == x.delta()

    def test_diff_product_rule(self):
        M = Matrix('M')
        x = Vector('x')
        d = (M * x).diff()
        assert isinstance(d, VAdd)
        assert d.N == 2
        assert isinstance(d.nodes[0], MVMul)
        assert isinstance(d.nodes[1], MVMul)
        assert d.nodes[0].left == M.diff()
        assert d.nodes[0].right == x
        assert d.nodes[1].left == M
        assert d.nodes[1].right == x.diff()


class TestMMMul:
    def test_creation(self):
        M, N = getMatrices('M N')
        result = MMMul(M, N)
        assert result.type == Expression.MATRIX

    def test_delta_product_rule(self):
        M, N = getMatrices('M N')
        d = (M * N).delta()
        assert isinstance(d, MAdd)
        assert d.N == 2
        assert isinstance(d.nodes[0], MMMul)
        assert isinstance(d.nodes[1], MMMul)

    def test_delta_product_rule_str(self):
        M, N = getMatrices('M N')
        d = (M * N).delta()
        assert str(d) == '(\\delta{M}N+M\\delta{N})'

    def test_delta_constant_left(self):
        J = Matrix('J', attr=['Constant'])
        M = Matrix('M')
        d = (J * M).delta()
        assert isinstance(d, MMMul)

    def test_diff_product_rule(self):
        M, N = getMatrices('M N')
        d = (M * N).diff()
        assert isinstance(d, MAdd)
        assert d.N == 2
        assert isinstance(d.nodes[0], MMMul)
        assert isinstance(d.nodes[1], MMMul)


class TestVVMul:
    def test_vecT_vec_is_scalar(self):
        x, y = getVectors(['x', 'y'])
        result = VVMul(Transpose(x), y)
        assert result.type == Expression.SCALAR

    def test_vec_vecT_is_matrix(self):
        x, y = getVectors(['x', 'y'])
        result = VVMul(x, Transpose(y))
        assert result.type == Expression.MATRIX

    def test_delta_product_rule_scalar(self):
        x, y = getVectors(['x', 'y'])
        d = VVMul(Transpose(x), y).delta()
        assert isinstance(d, Add)
        assert d.N == 2
        assert isinstance(d.nodes[0], VVMul)
        assert isinstance(d.nodes[1], VVMul)

    def test_delta_product_rule_scalar_str(self):
        x, y = getVectors(['x', 'y'])
        d = VVMul(Transpose(x), y).delta()
        assert str(d) == "((\\delta{x})'y+(x)'\\delta{y})"

    def test_delta_product_rule_matrix(self):
        x, y = getVectors(['x', 'y'])
        d = VVMul(x, Transpose(y)).delta()
        assert isinstance(d, MAdd)
        assert d.N == 2
        assert isinstance(d.nodes[0], VVMul)
        assert isinstance(d.nodes[1], VVMul)

    def test_delta_product_rule_matrix_str(self):
        x, y = getVectors(['x', 'y'])
        d = VVMul(x, Transpose(y)).delta()
        assert str(d) == "(\\delta{x}(y)'+x(\\delta{y})')"

    @pytest.mark.xfail(reason="Transpose does not propagate isConstant")
    def test_delta_both_constant(self):
        e1 = Vector('e1', attr=['Constant'])
        e2 = Vector('e2', attr=['Constant'])
        d = VVMul(Transpose(e1), e2).delta()
        assert d.isZero
