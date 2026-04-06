class Test(object):
    def __init__(self):
        self.val = None

    def __dot__(self, other):
        print("performing dot operation")


class DerivedClass(Test):
    def __init__(self):
        super(DerivedClass, self).__init__()


x1 = Test()
x2 = Test()

var = x1 == x2
y = DerivedClass()

verifybasedependency = isinstance(y, Test)
verifybasedependency2 = isinstance(x1, DerivedClass)
verifybasedependency3 = isinstance(x2, Test)

print("done")
