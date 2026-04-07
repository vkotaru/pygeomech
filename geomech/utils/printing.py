"""Printing utilities for expressions and equations of motion.

print_eom(eqns)     — display the EOM dict as LaTeX equations
print_tree(expr)    — display the expression tree structure for debugging
render_eom(eqns)    — compile EOM to PDF via pdflatex
"""

from __future__ import annotations

import os
import subprocess
import tempfile

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
        line = "\\int{" + lhs + " \\cdot " + "\\Big(" + rhs + "\\Big)}dt=0"
        print(line)


# ---------------------------------------------------------------------------
# Tree printer
# ---------------------------------------------------------------------------


def print_tree(expr, indent=0, label="", style="topdown"):
    """Print the expression tree structure for debugging.

    Each node shows its class name, type, and key properties.
    Children are indented below their parent.

    Parameters
    ----------
    style : str
        'topdown' (default) — graphical top-down tree with / and \\ branches.
        'indent' — original indented list format.
    """
    if style == "topdown":
        lines, _, _ = _render_topdown(expr)
        print("\n".join(lines))
    else:
        lines = []
        _build_tree(expr, lines, indent, label)
        print("\n".join(lines))
    print()
    print(f"expr: {repr(expr)}")


def tree_str(expr, indent=0, label="", style="topdown"):
    """Return the expression tree as a string (without printing)."""
    if style == "topdown":
        lines, _, _ = _render_topdown(expr)
        return "\n".join(lines)
    else:
        lines = []
        _build_tree(expr, lines, indent, label)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Human-readable repr
# ---------------------------------------------------------------------------


def repr_str(expr):
    """Return a human-readable string for the expression (used by __repr__)."""
    from geomech.core.operations.addition import Add, MAdd, VAdd
    from geomech.core.operations.calculus import TimeDerivative, TimeIntegral, Variation
    from geomech.core.operations.geometry import Cross, Dot, Hat, Transpose, Vee
    from geomech.core.operations.multiplication import MMMul, Mul, MVMul, SMMul, SVMul, VVMul

    nodes = getattr(expr, "nodes", None)

    # Leaf
    if nodes is None or len(nodes) == 0:
        name = getattr(expr, "name", None)
        if name is not None:
            return name
        value = getattr(expr, "value", None)
        if value is not None:
            return str(value)
        return "?"

    # N-ary addition
    if isinstance(expr, (Add, VAdd, MAdd)):
        parts = []
        for i, child in enumerate(nodes):
            s = repr_str(child)
            if i > 0 and not s.startswith("-"):
                parts.append(" + ")
            elif i > 0:
                parts.append(" ")
            parts.append(s)
        return "(" + "".join(parts) + ")"

    # Binary multiplication — check for negative scalar factor
    if isinstance(expr, (Mul, SVMul, SMMul, MVMul, MMMul, VVMul)):
        left = repr_str(expr.left)
        right = repr_str(expr.right)
        # Detect multiplying by -1
        lval = getattr(expr.left, "value", None)
        rval = getattr(expr.right, "value", None)
        if isinstance(lval, (int, float)) and lval == -1:
            return f"-{right}"
        if isinstance(rval, (int, float)) and rval == -1:
            return f"-{left}"
        return f"{left}*{right}"

    # Dot, Cross
    if isinstance(expr, Dot):
        return f"<{repr_str(expr.left)}, {repr_str(expr.right)}>"
    if isinstance(expr, Cross):
        return f"cross({repr_str(expr.left)}, {repr_str(expr.right)})"

    # Hat, Vee, Transpose
    if isinstance(expr, Hat):
        return f"hat({repr_str(expr.expr)})"
    if isinstance(expr, Vee):
        return f"vee({repr_str(expr.expr)})"
    if isinstance(expr, Transpose):
        return f"{repr_str(expr.expr)}'"

    # Calculus
    if isinstance(expr, Variation):
        return f"δ({repr_str(expr.expr)})"
    if isinstance(expr, TimeDerivative):
        inner = repr_str(expr.expr)
        # For simple names, put combining dot above: x → ẋ
        if inner.isalnum() and len(inner) <= 10:
            return inner + "\u0307"
        return f"d/dt({inner})"
    if isinstance(expr, TimeIntegral):
        return f"∫({repr_str(expr.expr)})dt"

    # Fallback
    return str(expr)


_SHORT_NAMES = {
    "TimeDerivative": "d/dt",
    "TimeIntegral": "∫dt",
    "Variation": "δ",
    "SVMul": "S*V",
    "SMMul": "S*M",
    "MVMul": "M*V",
    "MMMul": "M*M",
    "VVMul": "V*V",
}


def _node_label(expr, compact=False):
    """Get the display label for a node."""
    cls_name = type(expr).__name__
    nodes = getattr(expr, "nodes", None)
    if nodes is None or len(nodes) == 0:
        if compact:
            return _compact_leaf(expr)
        detail = _leaf_detail(expr)
        return f"{cls_name}({detail})"
    if compact:
        return _SHORT_NAMES.get(cls_name, cls_name)
    return cls_name


def _compact_leaf(expr):
    """Short leaf label: just name/value + type prefix + flag markers."""
    cls_name = type(expr).__name__
    name = getattr(expr, "name", None)
    value = getattr(expr, "value", None)
    flags = getattr(expr, "flags", None)

    # Type prefix
    prefix = ""
    if cls_name in ("Vector", "TS2", "TSO3"):
        prefix = "v:"
    elif cls_name in ("Matrix", "SkewSymmMatrix"):
        prefix = "M:"
    elif cls_name in ("S2",):
        prefix = "S2:"
    elif cls_name in ("SO3",):
        prefix = "SO3:"
    # Scalars get no prefix

    # Display name
    display = name if name is not None else str(value) if value is not None else "?"

    # Flag suffix: * = constant, numeric values shown inline
    suffix = ""
    if flags:
        is_const = getattr(flags, "is_constant", False)
        is_num = getattr(flags, "is_numeric", False)
        if is_num and value is not None:
            # Pure numeric: just show the value
            display = str(value)
            if is_const:
                suffix = "*"
        elif is_const:
            suffix = "*"

    return prefix + display + suffix


def _render_topdown(expr, gap=3):
    """Render expression as top-down tree with / and \\ branches.

    Returns (lines, root_col, width) where lines is a list of strings,
    root_col is the column of the root node center, and width is the
    total width of the rendered block.
    """
    label = _node_label(expr, compact=True)
    nodes = getattr(expr, "nodes", None)

    # Leaf
    if nodes is None or len(nodes) == 0:
        return [label], len(label) // 2, len(label)

    # Render children
    children = [_render_topdown(child, gap) for child in nodes]

    # Unary node — pipe connector
    if len(children) == 1:
        c_lines, c_root, c_width = children[0]
        label_start = max(0, c_root - len(label) // 2)
        width = max(c_width, label_start + len(label))
        lines = []
        lines.append(" " * label_start + label)
        lines.append(" " * c_root + "|")
        for line in c_lines:
            lines.append(line.ljust(width))
        return lines, label_start + len(label) // 2, width

    # Multiple children — place side by side with gap
    child_blocks = []
    total_width = 0
    for i, (c_lines, c_root, c_width) in enumerate(children):
        if i > 0:
            total_width += gap
        child_blocks.append((c_lines, c_root, c_width, total_width))
        total_width += c_width

    first_root = child_blocks[0][3] + child_blocks[0][1]
    last_root = child_blocks[-1][3] + child_blocks[-1][1]
    center = (first_root + last_root) // 2
    label_start = center - len(label) // 2

    # Shift everything right if label goes negative
    if label_start < 0:
        shift = -label_start
        label_start = 0
        child_blocks = [(cl, cr, cw, co + shift) for cl, cr, cw, co in child_blocks]
        total_width += shift
        first_root += shift
        last_root += shift

    width = max(total_width, label_start + len(label))
    parent_root = label_start + len(label) // 2

    lines = []
    # Parent label
    lines.append(" " * label_start + label)

    # Draw branch lines from parent down to each child
    # For 2 children, draw diagonal lines; for N, single connector row
    child_roots = [co + cr for (_, cr, _, co) in child_blocks]

    # Single connector row with / | \ pointing to each child
    connector = [" "] * width
    for cr in child_roots:
        if cr < parent_root:
            connector[cr] = "/"
        elif cr > parent_root:
            connector[cr] = "\\"
        else:
            connector[cr] = "|"
    lines.append("".join(connector).rstrip())

    # Merge child lines row by row
    max_child_lines = max(len(cb[0]) for cb in child_blocks)
    for row in range(max_child_lines):
        line = [" "] * width
        for c_lines_list, c_root, c_width, c_offset in child_blocks:
            if row < len(c_lines_list):
                text = c_lines_list[row]
                for j, ch in enumerate(text):
                    pos = c_offset + j
                    if pos < width and ch != " ":
                        line[pos] = ch
        lines.append("".join(line).rstrip())

    return lines, parent_root, width


def _build_tree(expr, lines, indent, label):
    """Recursively build tree lines."""
    prefix = "  " * indent
    connector = "├─ " if indent > 0 else ""
    tag = f"{label}: " if label else ""

    cls_name = type(expr).__name__
    nodes = getattr(expr, "nodes", None)

    # Leaf node — show name/value
    if nodes is None or len(nodes) == 0:
        detail = _leaf_detail(expr)
        lines.append(f"{prefix}{connector}{tag}{cls_name}({detail})")
        return

    # Internal node
    lines.append(f"{prefix}{connector}{tag}{cls_name}")

    for i, child in enumerate(nodes):
        child_label = _child_label(expr, i)
        _build_tree(child, lines, indent + 1, child_label)


def _leaf_detail(expr):
    """Format the detail string for a leaf node."""
    parts = []
    name = getattr(expr, "name", None)
    if name is not None:
        parts.append(f"'{name}'")
    value = getattr(expr, "value", None)
    if value is not None and not _is_default_value(value):
        parts.append(f"value={value}")
    flags = getattr(expr, "flags", None)
    if flags is not None:
        active = [
            f
            for f in [
                "is_constant",
                "is_zero",
                "is_ones",
                "is_unit_norm",
                "is_symmetric",
                "is_numeric",
                "is_manifold",
            ]
            if getattr(flags, f, False)
        ]
        if active:
            parts.append(", ".join(active))
    return ", ".join(parts)


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
    from geomech.core.operations.geometry import Cross, Dot
    from geomech.core.operations.multiplication import MMMul, Mul, MVMul, SMMul, SVMul, VVMul

    # Binary ops: label left/right
    if isinstance(expr, (Mul, SVMul, SMMul, MVMul, MMMul, VVMul, Dot, Cross)):
        return "L" if index == 0 else "R"

    # Unary ops: label as expr
    if hasattr(expr, "expr") and len(getattr(expr, "nodes", [])) == 1:
        return ""

    # N-ary: label with index
    return str(index)


# ---------------------------------------------------------------------------
# LaTeX conversion
# ---------------------------------------------------------------------------


def _collect_scalar_factors(expr):
    """Collect scalar multiplication chain into (sign, [factor_strings]).

    Walks nested Mul to collect all scalar factors.
    Returns (negative: bool, factors: list[str]).
    """
    from geomech.core.operations.multiplication import Mul

    if isinstance(expr, Mul):
        lneg, lfactors = _collect_scalar_factors(expr.left)
        rneg, rfactors = _collect_scalar_factors(expr.right)
        return (lneg ^ rneg, lfactors + rfactors)

    val = getattr(expr, "value", None)
    if isinstance(val, (int, float)):
        if val == -1 or val == -1.0:
            return (True, [])
        if val == 1 or val == 1.0:
            return (False, [])
        if val == 0.5:
            return (False, [r"\frac{1}{2}"])
        if val == -0.5:
            return (True, [r"\frac{1}{2}"])
        if val < 0:
            return (True, [str(abs(val))])
        return (False, [str(val)])

    return (False, [to_latex(expr)])


def to_latex(expr):
    """Convert an expression tree to a LaTeX string.

    Returns a string suitable for use with IPython.display.Math() or
    inside a LaTeX equation environment.
    """
    from geomech.core.operations.addition import Add, MAdd, VAdd
    from geomech.core.operations.calculus import TimeDerivative, TimeIntegral, Variation
    from geomech.core.operations.geometry import Cross, Dot, Hat, Transpose, Vee
    from geomech.core.operations.multiplication import MMMul, Mul, MVMul, SMMul, SVMul, VVMul

    nodes = getattr(expr, "nodes", None)

    # Leaf
    if nodes is None or len(nodes) == 0:
        name = getattr(expr, "name", None)
        if name is not None:
            val = getattr(expr, "value", None)
            if val is not None and isinstance(val, (int, float)):
                if val == -1 or val == -1.0:
                    return "-1"
                if val == 0.5:
                    return r"\frac{1}{2}"
                if val == -0.5:
                    return r"-\frac{1}{2}"
                if val == 1 or val == 1.0:
                    return "1"
                return str(val)
            return name
        return "?"

    # N-ary addition
    if isinstance(expr, (Add, VAdd, MAdd)):
        parts = []
        for i, child in enumerate(nodes):
            s = to_latex(child)
            if i > 0 and not s.startswith("-"):
                parts.append(" + ")
            elif i > 0:
                parts.append(" ")
            parts.append(s)
        return "\\left(" + "".join(parts) + "\\right)"

    # Scalar * Scalar — collect chain
    if isinstance(expr, Mul):
        neg, factors = _collect_scalar_factors(expr)
        result = " ".join(factors) if factors else "1"
        return f"-{result}" if neg else result

    # SVMul: vector * scalar → collect scalar, render as coeff * vector
    if isinstance(expr, SVMul):
        vec_str = to_latex(expr.left)
        neg, factors = _collect_scalar_factors(expr.right)
        if factors:
            coeff = " ".join(factors)
            result = f"{coeff} {vec_str}"
        else:
            result = vec_str
        return f"-{result}" if neg else result

    # SMMul: matrix * scalar
    if isinstance(expr, SMMul):
        mat_str = to_latex(expr.left)
        neg, factors = _collect_scalar_factors(expr.right)
        if factors:
            coeff = " ".join(factors)
            result = f"{coeff} {mat_str}"
        else:
            result = mat_str
        return f"-{result}" if neg else result

    # MVMul: matrix * vector
    if isinstance(expr, MVMul):
        return f"{to_latex(expr.left)} {to_latex(expr.right)}"

    # MMMul: matrix * matrix
    if isinstance(expr, MMMul):
        return f"{to_latex(expr.left)} {to_latex(expr.right)}"

    # VVMul: vector * vector^T
    if isinstance(expr, VVMul):
        return f"{to_latex(expr.left)} {to_latex(expr.right)}"

    # Dot product
    if isinstance(expr, Dot):
        return f"{to_latex(expr.left)} \\cdot {to_latex(expr.right)}"

    # Cross product
    if isinstance(expr, Cross):
        return f"{to_latex(expr.left)} \\times {to_latex(expr.right)}"

    # Hat (skew-symmetric)
    if isinstance(expr, Hat):
        return f"\\widehat{{{to_latex(expr.expr)}}}"

    # Vee
    if isinstance(expr, Vee):
        return f"\\left({to_latex(expr.expr)}\\right)^\\vee"

    # Transpose
    if isinstance(expr, Transpose):
        return f"{to_latex(expr.expr)}^T"

    # Time derivative — detect double derivative for ddot
    if isinstance(expr, TimeDerivative):
        if isinstance(expr.expr, TimeDerivative):
            return f"\\ddot{{{to_latex(expr.expr.expr)}}}"
        return f"\\dot{{{to_latex(expr.expr)}}}"

    # Variation
    if isinstance(expr, Variation):
        return f"\\delta {to_latex(expr.expr)}"

    # Time integral
    if isinstance(expr, TimeIntegral):
        return f"\\int {to_latex(expr.expr)} \\, dt"

    # Fallback
    return str(expr)


def display_latex(expr):
    """Display an expression as rendered LaTeX in a Jupyter notebook."""
    from IPython.display import Math, display

    display(Math(to_latex(expr)))


def _strip_outer_parens(s):
    """Remove \\left( ... \\right) wrapping if present at the outermost level."""
    if s.startswith("\\left(") and s.endswith("\\right)"):
        return s[6:-7]
    return s


def display_eom(eqns):
    """Display equations of motion as rendered LaTeX in a Jupyter notebook."""
    from IPython.display import Math, display

    for key, (var_vec, eqn) in eqns.items():
        latex_str = _strip_outer_parens(to_latex(eqn)) + " = 0"
        display(Math(latex_str))


def _latex_str_key(s):
    """Convert a raw str key (from EOM dicts) to cleaner LaTeX.

    Handles patterns like \\frac{d}{dt}(\\frac{d}{dt}(x)) → \\ddot{x}
    and \\frac{d}{dt}(\\omega_{q}) → \\dot{\\omega_{q}}.
    """
    import re

    # ddot: \frac{d}{dt}(\frac{d}{dt}(X)) → \ddot{X}
    m = re.match(r"^\\frac\{d\}\{dt\}\(\\frac\{d\}\{dt\}\((.+)\)\)$", s)
    if m:
        return f"\\ddot{{{m.group(1)}}}"
    # dot: \frac{d}{dt}(X) → \dot{X}
    m = re.match(r"^\\frac\{d\}\{dt\}\((.+)\)$", s)
    if m:
        return f"\\dot{{{m.group(1)}}}"
    return s


def display_standard_form(sf):
    """Display standard form as M*ddq + G*u + f = 0 in a Jupyter notebook."""
    from IPython.display import Math, display

    for key, eq in sf.items():
        parts = []
        # M * accel terms
        for accel, coeff in eq.M.items():
            coeff_str = to_latex(coeff)
            accel_str = _latex_str_key(accel)
            parts.append(f"{coeff_str} {accel_str}")
        # G * input terms
        for inp, coeff in eq.G.items():
            g_str = to_latex(coeff)
            if g_str in ("I", "1"):
                parts.append(inp)
            else:
                parts.append(f"{g_str} {inp}")
        # f term
        f_str = to_latex(eq.f)
        if f_str not in ("0v", "0"):
            parts.append(f_str)
        # Join with sign-aware concatenation
        if not parts:
            equation = "0 = 0"
        else:
            result = parts[0]
            for p in parts[1:]:
                if p.startswith("-"):
                    result += f" {p}"
                else:
                    result += f" + {p}"
            equation = result + " = 0"
        display(Math(f"{_latex_str_key(key)}: \\quad {equation}"))


# ---------------------------------------------------------------------------
# PDF renderer
# ---------------------------------------------------------------------------


def eom_to_latex(eqns):
    """Return a complete LaTeX document string for the EOM dict."""
    lines = [
        r"\documentclass[12pt]{article}",
        r"\usepackage{amsmath,amssymb}",
        r"\usepackage[margin=1in]{geometry}",
        r"\begin{document}",
        r"\section*{Equations of Motion}",
    ]
    for key, (var_vec, eqn) in eqns.items():
        lhs = str(var_vec)
        rhs = str(eqn)
        lines.append(r"\begin{equation}")
        lines.append(r"\int " + lhs + r" \cdot \Big(" + rhs + r"\Big)\, dt = 0")
        lines.append(r"\end{equation}")
    lines.append(r"\end{document}")
    return "\n".join(lines)


def render_eom(eqns, output="eom.pdf", open_pdf=True):
    """Compile EOM to PDF via pdflatex.

    Parameters
    ----------
    eqns : dict
        The dict returned by ``compute_eom``.
    output : str
        Output PDF path (default: ``eom.pdf`` in current directory).
    open_pdf : bool
        If True, open the PDF after compilation (macOS ``open``).
    """
    tex_src = eom_to_latex(eqns)
    output = os.path.abspath(output)
    pdf_name = os.path.splitext(os.path.basename(output))[0]

    with tempfile.TemporaryDirectory() as tmpdir:
        tex_path = os.path.join(tmpdir, pdf_name + ".tex")
        with open(tex_path, "w") as f:
            f.write(tex_src)

        result = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-output-directory", tmpdir, tex_path],
            capture_output=True,
            text=True,
        )
        pdf_tmp = os.path.join(tmpdir, pdf_name + ".pdf")
        if result.returncode != 0 or not os.path.exists(pdf_tmp):
            print("pdflatex failed:")
            print(result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)
            return None

        import shutil

        shutil.copy2(pdf_tmp, output)

    print(f"PDF written to {output}")
    if open_pdf:
        subprocess.run(["open", output])
    return output
