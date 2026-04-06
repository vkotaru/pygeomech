from geomech.core.base import Matrix, Scalar, Vector, getMatrices, getScalars, getVectors
from geomech.core.operations import Add, Cross, Dot, Hat, MAdd, MMMul, Mul, MVMul, SVMul, VAdd
from geomech.core.transformations import expand


class TestExpandScalar:
    def test_distribute_right(self):
        """a(b+c) = ab + ac"""
        a, b, c = getScalars("a b c")
        expr = a * (b + c)
        expanded = expand(expr)
        assert isinstance(expanded, Add)
        assert str(expanded) == "(ab+ac)"

    def test_distribute_left(self):
        """(a+b)c = ac + bc"""
        a, b, c = getScalars("a b c")
        expr = (a + b) * c
        expanded = expand(expr)
        assert isinstance(expanded, Add)
        assert str(expanded) == "(ac+bc)"

    def test_distribute_both(self):
        """(a+b)(c+d) = ac + ad + bc + bd"""
        a, b, c, d = getScalars("a b c d")
        expr = (a + b) * (c + d)
        expanded = expand(expr)
        assert isinstance(expanded, Add)
        assert len(expanded) == 4
        assert str(expanded) == "(ac+ad+bc+bd)"

    def test_no_change_simple(self):
        a, b = getScalars("a b")
        expr = a * b
        expanded = expand(expr)
        assert isinstance(expanded, Mul)
        assert str(expanded) == "ab"


class TestExpandDot:
    def test_distribute_right(self):
        """x.(u+v) = x.u + x.v"""
        x, u, v = getVectors(["x", "u", "v"])
        expr = Dot(x, u + v)
        expanded = expand(expr)
        assert isinstance(expanded, Add)
        assert str(expanded) == "(Dot(x,u)+Dot(x,v))"

    def test_distribute_left(self):
        """(x+y).u = x.u + y.u"""
        x, y, u = getVectors(["x", "y", "u"])
        expr = Dot(x + y, u)
        expanded = expand(expr)
        assert isinstance(expanded, Add)
        assert str(expanded) == "(Dot(x,u)+Dot(y,u))"

    def test_distribute_both(self):
        """(x+y).(u+v) = x.u + x.v + y.u + y.v"""
        x, y, u, v = getVectors(["x", "y", "u", "v"])
        expr = Dot(x + y, u + v)
        expanded = expand(expr)
        assert isinstance(expanded, Add)
        assert len(expanded) == 4
        assert str(expanded) == "(Dot(x,u)+Dot(x,v)+Dot(y,u)+Dot(y,v))"


class TestExpandVector:
    def test_distribute_mvmul_left(self):
        """(A+B)x = Ax+Bx"""
        A, B = getMatrices("A B")
        x = Vector("x")
        expr = MVMul(A + B, x)
        expanded = expand(expr)
        assert isinstance(expanded, VAdd)
        assert str(expanded) == "(Ax+Bx)"

    def test_distribute_mvmul_right(self):
        """A(x+y) = Ax + Ay"""
        A = Matrix("A")
        x, y = getVectors(["x", "y"])
        expr = MVMul(A, x + y)
        expanded = expand(expr)
        assert isinstance(expanded, VAdd)
        assert str(expanded) == "(Ax+Ay)"

    def test_distribute_svmul(self):
        """(x+y)a = xa + ya"""
        x, y = getVectors(["x", "y"])
        a = Scalar("a")
        expr = SVMul(x + y, a)
        expanded = expand(expr)
        assert isinstance(expanded, VAdd)
        assert str(expanded) == "(xa+ya)"

    def test_distribute_cross_right(self):
        x, y, z = getVectors(["x", "y", "z"])
        expr = Cross(x, y + z)
        expanded = expand(expr)
        assert isinstance(expanded, VAdd)

    def test_distribute_cross_left(self):
        x, y, z = getVectors(["x", "y", "z"])
        expr = Cross(x + y, z)
        expanded = expand(expr)
        assert isinstance(expanded, VAdd)


class TestExpandMatrix:
    def test_distribute_mmmul_left(self):
        """(A+B)C = AC+BC"""
        A, B, C = getMatrices("A B C")
        expr = MMMul(A + B, C)
        expanded = expand(expr)
        assert isinstance(expanded, MAdd)

    def test_distribute_mmmul_right(self):
        """A(B+C) = AB+AC"""
        A, B, C = getMatrices("A B C")
        expr = MMMul(A, B + C)
        expanded = expand(expr)
        assert isinstance(expanded, MAdd)

    def test_hat_expand(self):
        x, y = getVectors(["x", "y"])
        expr = Hat(x + y)
        expanded = expand(expr)
        # Hat is linear: Hat(x+y) = Hat(x) + Hat(y)
        assert isinstance(expanded, MAdd)
