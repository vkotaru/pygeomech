class ExpressionMismatchError(Exception):
    """Expression mismatch during the operation"""

    def __init__(self, operation="", ltype=None, rtype=None):
        if ltype is not None and rtype is not None:
            msg = operation + " cannot be performed between " + ltype.name + " and " + rtype.name
        else:
            msg = "operation cannot be performed"
        super().__init__(msg)


class SizeMismatchError(Exception):
    """Operand sizes do not match for the operation."""

    def __init__(self, operation="", expected=None, got=None):
        if expected is not None and got is not None:
            msg = operation + " size mismatch: expected " + str(expected) + ", got " + str(got)
        else:
            msg = operation + " operand sizes do not match"
        super().__init__(msg)


class AlgebraicError(Exception):
    """Expression cannot be algebraically factored as requested.

    Raised when an operation is mathematically impossible, not merely
    unimplemented — e.g. extracting a linear vector coefficient from
    an outer product (rank-1 matrix) that does not factor as C * q.
    """

    def __init__(self, msg=""):
        super().__init__(msg)


class NonLinearError(Exception):
    """Query appears nonlinearly (degree > 1) in the expression.

    Raised when linear coefficient extraction encounters the query
    in both children of a binary node — e.g. Mul(f(q), g(q)).
    """

    def __init__(self, msg=""):
        super().__init__(msg)


class UndefinedCaseError(Exception):
    """New case found"""

    def __init__(self):
        # TODO add operation and message types
        msg = "New (undefined) case has been found"
        super().__init__(msg)
