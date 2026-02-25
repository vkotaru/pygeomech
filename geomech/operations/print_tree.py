def print_latex(eqns):
    for eq in eqns.values():
        s = eq[1].__str__()
        if 'dot_dot_' in s:
            s = s.replace("dot_dot_", "\ddot ")
        if 'dot_' in s:
            s = s.replace("dot_", "\dot ")

        str = '\int{' + eq[0].__str__() + ' \cdot ' + '\Big(' + s + '\Big)}dt=0\n'
        print(str)
    return True
