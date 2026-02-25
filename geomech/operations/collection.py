from geomech.operations.multiplication import Mul, MVMul
from geomech.operations.addition import Add
from geomech.operations.geometry import Dot, Cross
from geomech.operations.binary_tree import is_member


def col(_scalar, _vector):
    """Collect scalar expression with respect to vector"""
    # Base Cases
    #
    # ADDITION
    if isinstance(_scalar, Add):
        collected_scalars = Add()
        for n in _scalar.nodes:
            collected_scalars += col(n, _vector)
        return collected_scalars

    #  MULTIPLICATION
    elif isinstance(_scalar, Mul):
        return col(_scalar.left, _vector) * col(_scalar.right, _vector)

    elif isinstance(_scalar, Dot):

        if _scalar.right == _vector:
            return Dot(_scalar.right, _scalar.left)

        elif _scalar.left == _vector:
            return _scalar

        elif not is_member(_scalar.left, _vector) and not is_member(_scalar.right, _vector):
            return _scalar

        elif isinstance(_scalar.right, MVMul):
            raise NotImplementedError

        elif isinstance(_scalar.left, MVMul):
            raise NotImplementedError
        #    // Potential Energy
        #    case Dot(u,VMul(MMul(MMul(r,SkewMat(s)),y),v)) =>
        #        if (Vec(s) == z) {Dot(Vec(s), Cross(VMul(transpose(r)***y,u),v))}
        #        else {Dot(u,VMul(MMul(MMul(r,SkewMat(s)),y),v))}
        #    case Dot(VMul(MMul(r,SkewMat(s)),v),u) =>
        #        if (Vec(s) == z) {Dot(Vec(s), Cross(VMul(transpose(r),u),v))}
        #        else {Dot(VMul(MMul(r,SkewMat(s)),v),u)}

        # KINETIC ENERGY
        elif isinstance(_scalar.left, Cross) and isinstance(_scalar.right, MVMul):
            return col(Dot(_scalar.right, _scalar.left), _vector)
        elif isinstance(_scalar.left, MVMul) and isinstance(_scalar.right, Cross):
            if _scalar.right.left == _vector:
                return Dot(_scalar.right.left, Cross(_scalar.right.right, _scalar.left))
            elif _scalar.right.right == _vector:
                return Dot(_scalar.right.right, Cross(_scalar.left, _scalar.right.left))
            elif _scalar.left.right == _vector:
                return Dot(_scalar.left.right, Cross(MVMul(_scalar.left.left, _scalar.right.left), _scalar.right.right))
            else:
                return _scalar

        elif isinstance(_scalar.left, MVMul):
            if _scalar.left.right == _vector:
                return Dot(_scalar.left.right, MVMul(_scalar.left.left, _scalar.right))
            else:
                return Dot(_scalar.right, _scalar.left)
        elif isinstance(_scalar.right, MVMul):
            return col(Dot(_scalar.right, _scalar.left), _vector)

        elif isinstance(_scalar.right, Cross):
            if _scalar.right.left == _vector:
                return Dot(_scalar.right.left, Cross(_scalar.right.right, _scalar.left))
            elif _scalar.right.right == _vector:
                return Dot(_scalar.right.right, Cross(_scalar.left, _scalar.right.left))
            else:
                return _scalar

        else:
            return _scalar

    else:
        return _scalar
