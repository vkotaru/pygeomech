"""Printing utilities for expressions and equations of motion.

print_eom(eqns)     — display the EOM dict as LaTeX equations
print_tree(expr)    — display the expression tree structure for debugging
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# EOM printer
# ---------------------------------------------------------------------------

def print_eom(eqns):
    """Print equations of motion as LaTeX integral equations.

    *eqns* is the dict returned by ``compute_eom``:
        {str(variation_vector): (variation_vector, equation)}
    """
    for key, (var_vec, eqn) in eqns.items():
        lhs = str(var_vec)
        rhs = str(eqn)
        line = '\\int{' + lhs + ' \\cdot ' + '\\Big(' + rhs + '\\Big)}dt=0'
        print(line)


# ---------------------------------------------------------------------------
# Tree printer
# ---------------------------------------------------------------------------

def print_tree(expr, indent=0, label=''):
    """Print the expression tree structure for debugging.

    Each node shows its class name, type, and key properties.
    Children are indented below their parent.
    """
    lines = []
    _build_tree(expr, lines, indent, label)
    print('\n'.join(lines))


def tree_str(expr, indent=0, label=''):
    """Return the expression tree as a string (without printing)."""
    lines = []
    _build_tree(expr, lines, indent, label)
    return '\n'.join(lines)


def _build_tree(expr, lines, indent, label):
    """Recursively build tree lines."""
    prefix = '  ' * indent
    connector = '├─ ' if indent > 0 else ''
    tag = f'{label}: ' if label else ''

    cls_name = type(expr).__name__
    nodes = getattr(expr, 'nodes', None)

    # Leaf node — show name/value
    if nodes is None or len(nodes) == 0:
        detail = _leaf_detail(expr)
        lines.append(f'{prefix}{connector}{tag}{cls_name}({detail})')
        return

    # Internal node
    lines.append(f'{prefix}{connector}{tag}{cls_name}')

    for i, child in enumerate(nodes):
        child_label = _child_label(expr, i)
        _build_tree(child, lines, indent + 1, child_label)


def _leaf_detail(expr):
    """Format the detail string for a leaf node."""
    parts = []
    name = getattr(expr, 'name', None)
    if name is not None:
        parts.append(f"'{name}'")
    value = getattr(expr, 'value', None)
    if value is not None and not _is_default_value(value):
        parts.append(f'value={value}')
    flags = getattr(expr, '_flags', None)
    if flags is not None:
        active = [f for f in ['is_constant', 'is_zero', 'is_ones', 'is_unit_norm',
                               'is_symmetric', 'is_numeric', 'is_manifold']
                  if getattr(flags, f, False)]
        if active:
            parts.append(', '.join(active))
    return ', '.join(parts)


def _is_default_value(value):
    """Check if a value is the default (array of None)."""
    try:
        import numpy as np
        if isinstance(value, np.ndarray) and all(v is None for v in value.flat):
            return True
    except (ImportError, TypeError):
        pass
    return False


def _child_label(expr, index):
    """Return a descriptive label for the i-th child of expr."""
    from geomech.core.operations.addition import Add, VAdd, MAdd
    from geomech.core.operations.multiplication import Mul, SVMul, SMMul, MVMul, MMMul, VVMul
    from geomech.core.operations.geometry import Dot, Cross

    # Binary ops: label left/right
    if isinstance(expr, (Mul, SVMul, SMMul, MVMul, MMMul, VVMul, Dot, Cross)):
        return 'L' if index == 0 else 'R'

    # Unary ops: label as expr
    if hasattr(expr, 'expr') and len(getattr(expr, 'nodes', [])) == 1:
        return ''

    # N-ary: label with index
    return str(index)
