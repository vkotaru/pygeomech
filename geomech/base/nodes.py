from abc import ABC


class Node(object):
    def __init__(self):
        super().__init__()

    def __str__(self):
        raise NotImplementedError

    def has(self, elem):
        raise NotImplementedError


class UnaryNode(Node, ABC):
    """
    Unary node used by transpose, hat, vee, and such operators
    """

    def __init__(self):
        super().__init__()
        self._expr = None

    @property
    def expr(self):
        return self._expr

    @expr.setter
    def expr(self, e):
        self._expr = e

    def has(self, elem):
        return elem.__str__() == self.__str__()


class BinaryNode(Node, ABC):
    """
    Binary Node used multiplication, dot, cross and other operators
    """

    def __init__(self, l=None, r=None):
        super().__init__()
        self.left = l
        self.right = r

    @property
    def left(self):
        return self._left

    @left.setter
    def left(self, l):
        self._left = l

    @property
    def right(self):
        return self._right

    @right.setter
    def right(self, r):
        self._right = r

    def has(self, elem):
        return self.left.has(elem) or self.right.has(elem)


class NaryNode(Node, ABC):
    """N-ary node for addition"""

    def __init__(self, *args):
        super().__init__()
        self._nodes = []
        for arg in args:
            if isinstance(arg, list) or isinstance(arg, tuple):
                for a in arg:
                    if a.type == self.type:
                        self.nodes.append(a)
                    else:
                        raise Exception('Input to ', type(self).__name__, 'should be Expression', str(self.type))
            else:
                if arg.type == self.type:
                    if isinstance(arg, type(self)):
                        self.nodes.extend(arg.nodes)
                    else:
                        self.nodes.append(arg)
                else:
                    raise Exception('Input to ', type(self).__name__, 'should be Expression', str(self.type))

    @property
    def nodes(self):
        return self._nodes

    @nodes.setter
    def nodes(self, n):
        self._nodes = n

    def has(self, elem):
        val = False
        for n in self.nodes:
            val = val or n.has(elem)
        return val

    def append(self, elem):
        self._nodes.append(elem)

    def replace_at(self, ind, elem):
        self.nodes[ind] = elem

    @property
    def N(self):
        """Number of nodes"""
        return len(self._nodes)

    def get_str(self, optr_sym='+'):
        s = '('
        for n in self.nodes[:-1]:
            s += n.__str__() + optr_sym
        s += self.nodes[-1].__str__() + ')'
        return s
