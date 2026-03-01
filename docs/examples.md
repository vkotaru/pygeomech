# Examples

## Point mass

Derives Newton's equation \(m\ddot{x} = f - mge_3\) for a particle in a gravitational field.

```python
from geomech import *
import numpy as np

m, g = getScalars('m g', attr=['Constant'])
e3 = Vector('e3', attr=['Constant'], value=np.array([0., 0., 1]))
x, f = getVectors(['x', 'f'])

v = x.t_diff()

KE = m * Dot(v, v) * 0.5
PE = m * x.dot(g * e3)
L  = KE - PE

dW = Dot(x.delta(), f)

eqs = compute_eom(L, dW, SystemVariables(vectors=[x]))
print_eom(eqs)
```

## Spherical pendulum on \(S^2\)

A point mass constrained to the surface of a sphere. The attitude \(q \in S^2\) is a unit vector -- no angles, no singularities.

```python
from geomech import *
import numpy as np

m, g, l = getScalars('m g l', attr=['Constant'])
e3 = Vector('e3', attr=['Constant'], value=np.array([0., 0., 1]))

q = S2('q')             # attitude on the 2-sphere
f = Vector('f')

x = l * q
v = x.t_diff()

KE = m * Dot(v, v) * 0.5
PE = m * x.dot(g * e3)
L  = KE - PE
dW = Dot(q.delta(), f)

eqs = compute_eom(L, dW, SystemVariables(vectors=[q]))
print_eom(eqs)
```

The library automatically uses the \(S^2\) variation formula \(\delta q = \xi \times q\) and time derivative \(\dot{q} = \omega \times q\) to produce the globally valid equations.

## Rigid body on \(SO(3)\)

A rigid body with rotation matrix \(R \in SO(3)\), body-frame inertia \(J\), and center of mass offset \(\rho\).

```python
from geomech import *
import numpy as np

J = Matrix('J', attr=['Constant', 'SymmetricMatrix'])
rho = Vector('\\rho', attr=['Constant'])
m, g = getScalars('m g', attr=['Constant'])
e3 = Vector('e_3', attr=['Constant'], value=np.array([0., 0., 1]))

R = SO3('R')             # rotation matrix on SO(3)
Om = R.get_tangent_vector()
eta = R.get_variation_vector()
M = Vector('M')          # external torque

x = R * rho
v = x.t_diff()

KE = Dot(Om, J * Om) * 0.5 + Dot(v, v) * m * 0.5
PE = m * g * Dot(x, e3)
L  = KE - PE
dW = Dot(eta, M)

eqs = compute_eom(L, dW, SystemVariables(matrices=[R]))
print_eom(eqs)
```

The \(SO(3)\) variation \(\delta R = R\hat{\eta}\) and kinematics \(\dot{R} = R\hat{\Omega}\) are applied automatically.

## Double rigid pendulum

Two rigid bodies connected in series -- demonstrates multi-body systems.

```python
from geomech import *
import numpy as np

J1 = Matrix('J1', attr=['Constant', 'SymmetricMatrix'])
J2 = Matrix('J2', attr=['Constant', 'SymmetricMatrix'])
rho1 = Vector('\\rho1', attr=['Constant'])
rho2 = Vector('\\rho2', attr=['Constant'])
l1 = Vector('l1', attr=['Constant'])
m1, m2, g = getScalars('m1 m2 g', attr=['Constant'])
e3 = Vector('e_3', attr=['Constant'], value=np.array([0., 0., 1]))

M1, M2 = getVectors('M1 M2')
R1, R2 = SO3('R1'), SO3('R2')
Om1, eta1 = R1.get_tangent_vector(), R1.get_variation_vector()
Om2, eta2 = R2.get_tangent_vector(), R2.get_variation_vector()

x1 = R1 * rho1
x2 = R1 * l1 + R2 * rho2
v2 = x2.t_diff()

KE = (Dot(Om1, J1 * Om1) * 0.5
    + Dot(Om2, J2 * Om2) * 0.5
    + Dot(v2, v2) * m2 * 0.5)
PE = m1 * g * Dot(R1 * rho1, e3) + m2 * g * Dot(R2 * rho2, e3)
L = KE - PE

dW = Dot(eta1, M1) + Dot(eta2, M2)

eqs = compute_eom(L, dW, SystemVariables(matrices=[R1, R2]))
print_eom(eqs)
```
