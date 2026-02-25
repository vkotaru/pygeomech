
class ExpressionMismatchError(Exception):
    """Expression mismatch during the operation"""
    def __init__(self, operation='', ltype=None, rtype=None):
        if ltype is not None and rtype is not None:
            msg =  operation + ' cannot be performed between ' + ltype.name + ' and ' + rtype.name
        else:
            msg = 'operation cannot be performed'
        super().__init__(msg)

class UndefinedCaseError(Exception):
    """New case found"""
    def __init__(self ):
        # TODO add operation and message types
        msg =  'New (undefined) case has been found'
        super().__init__(msg)