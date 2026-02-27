"""Coefficient extraction: factor a target vector out of an expression.

  extract_from_scalar — coefficient of vec in a scalar expression
  extract_from_vector — coefficient of vec in a vector expression
  extract_from_matrix — coefficient of vec in a matrix expression (partial)

extract_coeff(expr, vec) dispatches based on expr.type.

Assumes *vec* appears linearly (degree 1) in each term.  Expressions
where *vec* appears on both sides of a product (e.g. Dot(v, v), Mul
with vec in both operands) raise NotImplementedError.
"""

from __future__ import annotations

from geomech.core.base.expressions import ZeroVector, ZeroMatrix
from geomech.core.base.types import ExprType
from geomech.core.operations.addition import Add, VAdd, MAdd
from geomech.core.operations.multiplication import (
    Mul, SVMul, SMMul, MVMul, MMMul, VVMul,
)
from geomech.core.operations.geometry import Dot, Cross, Hat, Transpose


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_coeff(expr, vec):
    """Extract the coefficient of *vec* from *expr*.

    Dispatches to extract_from_scalar / extract_from_vector / extract_from_matrix based on expr.type.
    """
    match getattr(expr, 'type', None):
        case ExprType.SCALAR:
            return extract_from_scalar(expr, vec)
        case ExprType.VECTOR:
            return extract_from_vector(expr, vec)
        case ExprType.MATRIX:
            return extract_from_matrix(expr, vec)
        case _:
            raise NotImplementedError(
                f"extract_coeff: unsupported type {getattr(expr, 'type', None)}"
            )


# ---------------------------------------------------------------------------
# extract_from_scalar
# ---------------------------------------------------------------------------

def extract_from_scalar(expr, vec):
    """Extract the coefficient of *vec* from a scalar expression."""
    match expr:
        # --- scalar addition: linearity ---
        case Add(nodes=nodes):
            extracted = [extract_from_scalar(n, vec) for n in nodes if n.has(vec)]
            if not extracted:
                return ZeroVector
            if len(extracted) == 1:
                return extracted[0]
            return VAdd(*extracted)

        # --- scalar multiplication ---
        case Mul():
            l, r = expr.left, expr.right
            if l.has(vec) and r.has(vec):
                raise NotImplementedError("extract_from_scalar: both sides of Mul contain vec")
            if l.has(vec):
                return SVMul(extract_from_scalar(l, vec), r)
            if r.has(vec):
                return SVMul(extract_from_scalar(r, vec), l)
            return ZeroVector

        # --- dot product: core extraction ---
        case Dot():
            l, r = expr.left, expr.right
            if l.has(vec) and r.has(vec):
                raise NotImplementedError("extract_from_scalar: both sides of Dot contain vec")
            if l.has(vec):
                if l == vec:
                    return r
                if l.type == ExprType.VECTOR:
                    return MVMul(Transpose(extract_from_vector(l, vec)), r)
                raise NotImplementedError(
                    f"extract_from_scalar: Dot.left contains vec but type={l.type}"
                )
            if r.has(vec):
                if r == vec:
                    return l
                if r.type == ExprType.VECTOR:
                    # Dot(l, M*vec) = l^T M vec = (M^T l)^T vec = Dot(vec, M^T l)
                    return MVMul(Transpose(extract_from_vector(r, vec)), l)
                raise NotImplementedError(
                    f"extract_from_scalar: Dot.right contains vec but type={r.type}"
                )
            return ZeroVector

        case VVMul():
            raise NotImplementedError("extract_from_scalar: VVMul")

        # --- leaf / unhandled ---
        case _:
            return ZeroVector


# ---------------------------------------------------------------------------
# extract_from_vector
# ---------------------------------------------------------------------------

def extract_from_vector(expr, vec):
    """Extract the coefficient of *vec* from a vector expression."""
    match expr:
        # --- vector addition: linearity ---
        case VAdd(nodes=nodes):
            extracted = [extract_from_vector(n, vec) for n in nodes if n.has(vec)]
            if not extracted:
                return ZeroMatrix
            if len(extracted) == 1:
                return extracted[0]
            return MAdd(*extracted)

        # --- cross product ---
        case Cross():
            l, r = expr.left, expr.right
            if l == r:
                return ZeroMatrix
            # cross(a, vec) = Hat(a) * vec
            if r == vec:
                return Hat(l)
            # cross(vec, b) = -Hat(b) * vec
            if l == vec:
                return SMMul(Hat(r), -1)
            if l.has(vec) and r.has(vec):
                raise NotImplementedError("extract_from_vector: both sides of Cross contain vec")
            # cross(f(vec), b) = -Hat(b) * f(vec) = -Hat(b) * M * vec
            if l.has(vec):
                return MMMul(SMMul(Hat(r), -1), extract_from_vector(l, vec))
            # cross(a, g(vec)) = Hat(a) * g(vec) = Hat(a) * M * vec
            if r.has(vec):
                return MMMul(Hat(l), extract_from_vector(r, vec))
            return ZeroMatrix

        # --- matrix * vector ---
        case MVMul():
            mat, v = expr.left, expr.right
            if v.has(vec):
                if v == vec:
                    return mat
                # M * f(vec) = M * N * vec
                return MMMul(mat, extract_from_vector(v, vec))
            if mat.has(vec):
                return _extract_vec_mvmul_mat(mat, v, vec)
            return ZeroMatrix

        # --- scalar * vector ---
        case SVMul():
            v, s = expr.left, expr.right
            if v.has(vec) and s.has(vec):
                raise NotImplementedError("extract_from_vector: both sides of SVMul contain vec")
            if v.has(vec):
                # s * f(vec) = s * M * vec = (s*M) * vec
                return SMMul(extract_from_vector(v, vec), s)
            if s.has(vec):
                raise NotImplementedError("extract_from_vector: SVMul scalar contains vec")
            return ZeroMatrix

        # --- leaf / unhandled ---
        case _:
            raise NotImplementedError(f"extract_from_vector: unhandled {type(expr).__name__}")


def _extract_vec_mvmul_mat(mat, b, vec):
    """Handle extract_from_vector(MVMul(mat, b), vec) where *mat* contains vec."""
    match mat:
        case MAdd(nodes=nodes):
            extracted = [extract_from_vector(MVMul(n, b), vec) for n in nodes if n.has(vec)]
            if not extracted:
                return ZeroMatrix
            if len(extracted) == 1:
                return extracted[0]
            return MAdd(*extracted)

        case MMMul():
            A, B = mat.left, mat.right
            # (A * B) * b → A * (B * b)
            if A.has(vec):
                return extract_from_vector(MVMul(A, MVMul(B, b)), vec)
            if B.has(vec):
                return MMMul(A, extract_from_vector(MVMul(B, b), vec))
            return ZeroMatrix

        case SMMul():
            M, s = mat.left, mat.right
            # (s * M) * b → M * (s * b)
            if M.has(vec):
                return extract_from_vector(MVMul(M, SVMul(b, s)), vec)
            if s.has(vec):
                raise NotImplementedError(
                    "extract_from_vector: SMMul scalar contains vec in MVMul context"
                )
            return ZeroMatrix

        case Hat():
            # Hat(vec) * b = cross(vec, b) = -Hat(b) * vec → -Hat(b)
            if mat.expr == vec:
                return SMMul(Hat(b), -1)
            # Hat(f(vec)) * b = -Hat(b) * f(vec) = -Hat(b) * M * vec
            if mat.expr.has(vec):
                return MMMul(SMMul(Hat(b), -1), extract_from_vector(mat.expr, vec))
            return ZeroMatrix

        case VVMul():
            raise NotImplementedError("extract_from_vector: VVMul in MVMul.left")

        case _:
            return ZeroMatrix


# ---------------------------------------------------------------------------
# extract_from_matrix
# ---------------------------------------------------------------------------

def extract_from_matrix(expr, vec):
    """Extract the coefficient of *vec* from a matrix expression.

    Partial — only limited cases supported.
    """
    match expr:
        case MAdd(nodes=nodes):
            extracted = [extract_from_matrix(n, vec) for n in nodes if n.has(vec)]
            if not extracted:
                return ZeroMatrix
            if len(extracted) == 1:
                return extracted[0]
            return MAdd(*extracted)

        case MMMul():
            l, r = expr.left, expr.right
            if l.has(vec) and r.has(vec):
                raise NotImplementedError("extract_from_matrix: both sides of MMMul contain vec")
            if l.has(vec):
                raise NotImplementedError("extract_from_matrix: MMMul.left contains vec")
            if r.has(vec):
                if r == vec:
                    return l
                raise NotImplementedError("extract_from_matrix: MMMul.right non-leaf contains vec")
            return ZeroMatrix

        case VVMul():
            raise NotImplementedError("extract_from_matrix: VVMul")

        case SMMul():
            raise NotImplementedError("extract_from_matrix: SMMul")

        case _:
            raise NotImplementedError(f"extract_from_matrix: unhandled {type(expr).__name__}")
