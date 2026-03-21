from geomech.core.base.expressions import (
    Scalar, Vector, Matrix, getVectors, getMatrices,
)
from geomech.core.operations.addition import Add, VAdd, MAdd
from geomech.core.operations.multiplication import Mul, SVMul, SMMul, MVMul, MMMul
from geomech.core.operations.geometry import Dot, Cross, Hat, Vee
from geomech.core.transformations.expand import expand


# ---------------------------------------------------------------------------
# Leaf passthrough
# ---------------------------------------------------------------------------

class TestLeafPassthrough:
    def test_scalar_leaf(self):
        """Scalar('a') is a leaf — returned as-is."""
        a = Scalar('a')
        assert expand(a) is a

    def test_vector_leaf(self):
        """Vector('x') is a leaf — returned as-is."""
        x = Vector('x')
        assert expand(x) is x

    def test_matrix_leaf(self):
        """Matrix('M') is a leaf — returned as-is."""
        M = Matrix('M')
        assert expand(M) is M


# ---------------------------------------------------------------------------
# Addition recurse
# ---------------------------------------------------------------------------

class TestAddRecurse:
    def test_scalar_add_recurses(self):
        """expand recurses into each term of a scalar sum."""
        a, b = Scalar('a'), Scalar('b')
        result = expand(Add(a, b))
        assert isinstance(result, Add)
        assert len(result.nodes) == 2

    def test_vector_add_recurses(self):
        """expand recurses into each term of a vector sum."""
        x, y = getVectors(['x', 'y'])
        result = expand(VAdd(x, y))
        assert isinstance(result, VAdd)
        assert len(result.nodes) == 2

    def test_matrix_add_recurses(self):
        """expand recurses into each term of a matrix sum."""
        A, B = Matrix('A'), Matrix('B')
        result = expand(MAdd(A, B))
        assert isinstance(result, MAdd)
        assert len(result.nodes) == 2

    def test_add_expands_nested_mul(self):
        """Add(Mul(Add(a,b), c), d) → Add(a*c, b*c, d) via flattening."""
        a, b, c, d = [Scalar(n) for n in 'abcd']
        result = expand(Add(Mul(Add(a, b), c), d))
        assert isinstance(result, Add)
        assert len(result.nodes) == 3


# ---------------------------------------------------------------------------
# Mul distributes over Add
# ---------------------------------------------------------------------------

class TestMulDistribute:
    def test_mul_add_left(self):
        """(a+b)*c → a*c + b*c — distribute left."""
        a, b, c = [Scalar(n) for n in 'abc']
        result = expand(Mul(Add(a, b), c))
        assert isinstance(result, Add)
        assert len(result.nodes) == 2
        assert all(isinstance(n, Mul) for n in result.nodes)

    def test_mul_add_right(self):
        """a*(b+c) → a*b + a*c — distribute right."""
        a, b, c = [Scalar(n) for n in 'abc']
        result = expand(Mul(a, Add(b, c)))
        assert isinstance(result, Add)
        assert len(result.nodes) == 2

    def test_mul_add_both(self):
        """(a+b)*(c+d) → a*c + a*d + b*c + b*d — cartesian product."""
        a, b, c, d = [Scalar(n) for n in 'abcd']
        result = expand(Mul(Add(a, b), Add(c, d)))
        assert isinstance(result, Add)
        assert len(result.nodes) == 4

    def test_mul_no_add(self):
        """a*b unchanged when neither side is a sum."""
        a, b = Scalar('a'), Scalar('b')
        result = expand(Mul(a, b))
        assert isinstance(result, Mul)

    def test_mul_nested_add(self):
        """Mul(Mul(Add(a,b), c), d) — bottom-up expands inner first.

        Inner: (a+b)*c → Add(a*c, b*c)
        Outer: Add(a*c, b*c) * d → Add(a*c*d, b*c*d)
        """
        a, b, c, d = [Scalar(n) for n in 'abcd']
        result = expand(Mul(Mul(Add(a, b), c), d))
        assert isinstance(result, Add)
        assert len(result.nodes) == 2
        assert all(isinstance(n, Mul) for n in result.nodes)


# ---------------------------------------------------------------------------
# Dot distributes over VAdd
# ---------------------------------------------------------------------------

class TestDotDistribute:
    def test_dot_vadd_right(self):
        """dot(x, u+v) → dot(x,u) + dot(x,v) — distribute right."""
        x, u, v = getVectors(['x', 'u', 'v'])
        result = expand(Dot(x, VAdd(u, v)))
        assert isinstance(result, Add)
        assert len(result.nodes) == 2
        assert all(isinstance(n, Dot) for n in result.nodes)

    def test_dot_vadd_left(self):
        """dot(x+y, u) → dot(x,u) + dot(y,u) — distribute left."""
        x, y, u = getVectors(['x', 'y', 'u'])
        result = expand(Dot(VAdd(x, y), u))
        assert isinstance(result, Add)
        assert len(result.nodes) == 2

    def test_dot_vadd_both(self):
        """dot(x+y, u+v) → sum of four dot products."""
        x, y = getVectors(['x', 'y'])
        u, v = getVectors(['u', 'v'])
        result = expand(Dot(VAdd(x, y), VAdd(u, v)))
        assert isinstance(result, Add)
        assert len(result.nodes) == 4

    def test_dot_no_add(self):
        """dot(x, y) unchanged when no addition."""
        x, y = getVectors(['x', 'y'])
        result = expand(Dot(x, y))
        assert isinstance(result, Dot)


# ---------------------------------------------------------------------------
# Cross distributes over VAdd
# ---------------------------------------------------------------------------

class TestCrossDistribute:
    def test_cross_vadd_right(self):
        """cross(x, u+v) → cross(x,u) + cross(x,v) — distribute right."""
        x, u, v = getVectors(['x', 'u', 'v'])
        result = expand(Cross(x, VAdd(u, v)))
        assert isinstance(result, VAdd)
        assert len(result.nodes) == 2
        assert all(isinstance(n, Cross) for n in result.nodes)

    def test_cross_vadd_left(self):
        """cross(x+y, u) → cross(x,u) + cross(y,u) — distribute left."""
        x, y, u = getVectors(['x', 'y', 'u'])
        result = expand(Cross(VAdd(x, y), u))
        assert isinstance(result, VAdd)
        assert len(result.nodes) == 2

    def test_cross_vadd_both(self):
        """cross(x+y, u+v) → sum of four cross products."""
        x, y = getVectors(['x', 'y'])
        u, v = getVectors(['u', 'v'])
        result = expand(Cross(VAdd(x, y), VAdd(u, v)))
        assert isinstance(result, VAdd)
        assert len(result.nodes) == 4

    def test_cross_no_add(self):
        """cross(x, y) unchanged when no addition."""
        x, y = getVectors(['x', 'y'])
        result = expand(Cross(x, y))
        assert isinstance(result, Cross)


# ---------------------------------------------------------------------------
# SVMul distributes over VAdd / Add
# ---------------------------------------------------------------------------

class TestSVMulDistribute:
    def test_svmul_vadd_left(self):
        """(x+y)*a → x*a + y*a — VAdd on vector side."""
        x, y = getVectors(['x', 'y'])
        a = Scalar('a')
        result = expand(SVMul(VAdd(x, y), a))
        assert isinstance(result, VAdd)
        assert len(result.nodes) == 2
        assert all(isinstance(n, SVMul) for n in result.nodes)

    def test_svmul_add_right(self):
        """v*(a+b) → v*a + v*b — Add on scalar side."""
        v = Vector('v')
        a, b = Scalar('a'), Scalar('b')
        result = expand(SVMul(v, Add(a, b)))
        assert isinstance(result, VAdd)
        assert len(result.nodes) == 2

    def test_svmul_both(self):
        """(x+y)*(a+b) → four SVMul terms — cartesian product."""
        x, y = getVectors(['x', 'y'])
        a, b = Scalar('a'), Scalar('b')
        result = expand(SVMul(VAdd(x, y), Add(a, b)))
        assert isinstance(result, VAdd)
        assert len(result.nodes) == 4

    def test_svmul_no_add(self):
        """v*a unchanged when no addition on either side."""
        v = Vector('v')
        a = Scalar('a')
        result = expand(SVMul(v, a))
        assert isinstance(result, SVMul)


# ---------------------------------------------------------------------------
# MVMul distributes over MAdd / VAdd
# ---------------------------------------------------------------------------

class TestMVMulDistribute:
    def test_mvmul_madd_left(self):
        """(A+B)*x → A*x + B*x — MAdd on matrix side."""
        A, B = Matrix('A'), Matrix('B')
        x = Vector('x')
        result = expand(MVMul(MAdd(A, B), x))
        assert isinstance(result, VAdd)
        assert len(result.nodes) == 2
        assert all(isinstance(n, MVMul) for n in result.nodes)

    def test_mvmul_vadd_right(self):
        """A*(x+y) → A*x + A*y — VAdd on vector side."""
        A = Matrix('A')
        x, y = getVectors(['x', 'y'])
        result = expand(MVMul(A, VAdd(x, y)))
        assert isinstance(result, VAdd)
        assert len(result.nodes) == 2

    def test_mvmul_both(self):
        """(A+B)*(x+y) → four MVMul terms — cartesian product."""
        A, B = Matrix('A'), Matrix('B')
        x, y = getVectors(['x', 'y'])
        result = expand(MVMul(MAdd(A, B), VAdd(x, y)))
        assert isinstance(result, VAdd)
        assert len(result.nodes) == 4

    def test_mvmul_no_add(self):
        """A*x unchanged when no addition on either side."""
        A = Matrix('A')
        x = Vector('x')
        result = expand(MVMul(A, x))
        assert isinstance(result, MVMul)


# ---------------------------------------------------------------------------
# SMMul distributes over MAdd / Add
# ---------------------------------------------------------------------------

class TestSMMulDistribute:
    def test_smmul_madd_left(self):
        """(A+B)*s → A*s + B*s — MAdd on matrix side."""
        A, B = Matrix('A'), Matrix('B')
        s = Scalar('s')
        result = expand(SMMul(MAdd(A, B), s))
        assert isinstance(result, MAdd)
        assert len(result.nodes) == 2
        assert all(isinstance(n, SMMul) for n in result.nodes)

    def test_smmul_add_right(self):
        """M*(a+b) → M*a + M*b — Add on scalar side."""
        M = Matrix('M')
        a, b = Scalar('a'), Scalar('b')
        result = expand(SMMul(M, Add(a, b)))
        assert isinstance(result, MAdd)
        assert len(result.nodes) == 2

    def test_smmul_both(self):
        """(A+B)*(a+b) → four SMMul terms — cartesian product."""
        A, B = Matrix('A'), Matrix('B')
        a, b = Scalar('a'), Scalar('b')
        result = expand(SMMul(MAdd(A, B), Add(a, b)))
        assert isinstance(result, MAdd)
        assert len(result.nodes) == 4

    def test_smmul_no_add(self):
        """M*a unchanged when no addition on either side."""
        M = Matrix('M')
        a = Scalar('a')
        result = expand(SMMul(M, a))
        assert isinstance(result, SMMul)


# ---------------------------------------------------------------------------
# MMMul distributes over MAdd
# ---------------------------------------------------------------------------

class TestMMMulDistribute:
    def test_mmmul_madd_left(self):
        """(A+B)*C → A*C + B*C — distribute left."""
        A, B, C = Matrix('A'), Matrix('B'), Matrix('C')
        result = expand(MMMul(MAdd(A, B), C))
        assert isinstance(result, MAdd)
        assert len(result.nodes) == 2
        assert all(isinstance(n, MMMul) for n in result.nodes)

    def test_mmmul_madd_right(self):
        """A*(B+C) → A*B + A*C — distribute right."""
        A, B, C = Matrix('A'), Matrix('B'), Matrix('C')
        result = expand(MMMul(A, MAdd(B, C)))
        assert isinstance(result, MAdd)
        assert len(result.nodes) == 2

    def test_mmmul_madd_both(self):
        """(A+B)*(C+D) → four MMMul terms — cartesian product."""
        A, B, C, D = [Matrix(n) for n in ['A', 'B', 'C', 'D']]
        result = expand(MMMul(MAdd(A, B), MAdd(C, D)))
        assert isinstance(result, MAdd)
        assert len(result.nodes) == 4

    def test_mmmul_no_add(self):
        """A*B unchanged when no addition on either side."""
        A, B = Matrix('A'), Matrix('B')
        result = expand(MMMul(A, B))
        assert isinstance(result, MMMul)


# ---------------------------------------------------------------------------
# Unary ops recurse into child
# ---------------------------------------------------------------------------

class TestUnaryRecurse:
    def test_hat_recurses(self):
        """Hat(x) — no change for leaf vector, but recurse happens."""
        x = Vector('x')
        result = expand(Hat(x))
        assert isinstance(result, Hat)

    def test_vee_recurses(self):
        """Vee(M) — no change for leaf matrix, but recurse happens."""
        M = Matrix('M')
        result = expand(Vee(M))
        assert isinstance(result, Vee)

    def test_hat_with_nested_expansion(self):
        """Hat distributes over VAdd: Hat(x+y) → MAdd(Hat(x), Hat(y))."""
        x, y = getVectors(['x', 'y'])
        result = expand(Hat(VAdd(x, y)))
        assert isinstance(result, MAdd)
        assert len(result.nodes) == 2
        assert all(isinstance(n, Hat) for n in result.nodes)


# ---------------------------------------------------------------------------
# Nested / multi-level expansion
# ---------------------------------------------------------------------------

class TestNestedExpansion:
    def test_dot_with_svmul_vadd(self):
        """dot(SVMul(VAdd(x,y), a), z) — bottom-up expands SVMul first.

        SVMul(VAdd(x,y), a) → VAdd(SVMul(x,a), SVMul(y,a))
        Then: Dot(VAdd(...), z) → Add(Dot(SVMul(x,a), z), Dot(SVMul(y,a), z))
        """
        x, y, z = getVectors(['x', 'y', 'z'])
        a = Scalar('a')
        result = expand(Dot(SVMul(VAdd(x, y), a), z))
        assert isinstance(result, Add)
        assert len(result.nodes) == 2

    def test_mul_with_dot_vadd(self):
        """Mul(Dot(VAdd(x,y), z), a) — Dot distributes, then Mul wraps.

        Dot(VAdd(x,y), z) → Add(Dot(x,z), Dot(y,z))
        Then: Mul(Add(...), a) → Add(Mul(Dot(x,z), a), Mul(Dot(y,z), a))
        """
        x, y, z = getVectors(['x', 'y', 'z'])
        a = Scalar('a')
        result = expand(Mul(Dot(VAdd(x, y), z), a))
        assert isinstance(result, Add)
        assert len(result.nodes) == 2
        assert all(isinstance(n, Mul) for n in result.nodes)

    def test_cross_with_mvmul_madd(self):
        """cross(MVMul(MAdd(A,B), x), y) — MVMul distributes first.

        MVMul(MAdd(A,B), x) → VAdd(MVMul(A,x), MVMul(B,x))
        Then: Cross(VAdd(...), y) → VAdd(Cross(MVMul(A,x), y), Cross(MVMul(B,x), y))
        """
        A, B = Matrix('A'), Matrix('B')
        x, y = getVectors(['x', 'y'])
        result = expand(Cross(MVMul(MAdd(A, B), x), y))
        assert isinstance(result, VAdd)
        assert len(result.nodes) == 2
        assert all(isinstance(n, Cross) for n in result.nodes)

    def test_three_term_add_in_mul(self):
        """(a+b+c)*d → a*d + b*d + c*d — three-term distribution."""
        a, b, c, d = [Scalar(n) for n in 'abcd']
        result = expand(Mul(Add(a, b, c), d))
        assert isinstance(result, Add)
        assert len(result.nodes) == 3

    def test_deeply_nested(self):
        """Mul(Mul(Mul(Add(a,b), c), d), e) — three levels deep.

        Innermost: (a+b)*c → Add(a*c, b*c)
        Middle: Add(a*c, b*c)*d → Add(a*c*d, b*c*d)
        Outer: Add(a*c*d, b*c*d)*e → Add(a*c*d*e, b*c*d*e)
        """
        a, b, c, d, e = [Scalar(n) for n in 'abcde']
        result = expand(Mul(Mul(Mul(Add(a, b), c), d), e))
        assert isinstance(result, Add)
        assert len(result.nodes) == 2
