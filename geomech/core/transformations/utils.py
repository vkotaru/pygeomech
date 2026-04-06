from geomech.core.operations.addition import Add, MAdd, VAdd
from geomech.core.operations.mixins import _BinaryMixin, _NaryMixin


def is_leaf(expr):
    match expr:
        case _BinaryMixin() | _NaryMixin():
            return False
        case _:
            return True


def has_nested_add(expr):
    match expr:
        case Add() | VAdd() | MAdd():
            return True
        case _BinaryMixin():
            return has_nested_add(expr.left) or has_nested_add(expr.right)
        case _:
            return False
