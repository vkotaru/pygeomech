"""
expr.py defines the Expr class
"""
from enum import Enum


class Expression(Enum):
    SCALAR = 1
    VECTOR = 2
    MATRIX = 3


class Expr(object):
    """
    Expr (expression) base-class with the default properties and methods
    """

    def __init__(self):
        # name of the
        self._name = None
        self._value = None
        self._size = None
        self._type = None
        self._attr = None

        # expr attributes
        self.isConstant = False
        self.isZero = False
        self.isOnes = False
        self.isUnitNorm = False  # TODO add derivation dependency  # TODO add derivation level
        self.isManifold = False
        self.isSymmetric = False
        self.isNumeric = False

        self.derv_num = 0 # derivative level # TODO

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, s):
        self._name = s

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, val):
        self._value = val
        if val is not None:
            self.isNumeric = True

    @property
    def size(self):
        return self._size

    @size.setter
    def size(self, value):
        self._size = value

    @property
    def attr(self):
        return self._attr

    @attr.setter
    def attr(self, attributes):
        if attributes is not None:
            self._attr = attributes
            if 'Zero' in attributes:
                self.isZero = True
                self.isConstant = True
            else:
                self.isZero = False
            if 'Ones' in attributes:
                self.isOnes = True
                self.isConstant = True
            else:
                self.isOnes = False
            if 'Constant' in attributes:
                self.isConstant = True
            if 'Manifold' in attributes:
                self.isManifold = True
        else:
            self._attr = []

    @property
    def type(self):
        return self._type

    @type.setter
    def type(self, expr_type):
        self._type = expr_type

    def __str__(self):
        raise NotImplementedError

    def __eq__(self, other):
        return self.__str__() == other.__str__()

    def __add__(self, other):
        raise NotImplementedError

    def __mul__(self, other):
        raise NotImplementedError

    def __sub__(self, other):
        raise NotImplementedError

    def delta(self):
        raise NotImplementedError

    def diff(self):
        raise NotImplementedError

    def integrate(self):
        raise NotImplementedError

    def replace(self, old, new):
        self._name.replace(old, new)

    def flatten(self):
        raise NotImplementedError

    def simplify(self):
        raise NotImplementedError

    def variation_vector(self):
        raise NotImplementedError

    def has(self, elem):
        return elem.__str__() == self.__str__()


class Manifold(object):
    def __init__(self):
        self.isManifold = True
        self.tangent_vector = None
        self.variation_vector = None

    def get_variation_vector(self):
        raise NotImplementedError

    def get_tangent_vector(self):
        raise NotImplementedError
