# geomech

A Python library for symbolic derivation of equations of motion, variation based linearization on smooth manifolds, $$R,R^3,SO(3),S^2,SE(3)$$.

This work implements the methods described in [Global Formulations of Lagrangian and Hamiltonian Dynamics on Manifolds](https://link.springer.com/book/10.1007/978-3-319-56953-6) and extends the work developed in
[Kotaru & Sreenath, 2020](https://link.springer.com/chapter/10.1007/978-3-030-43089-4_22) and its [Scala implementation](https://github.com/HybridRobotics/dynamics_on_manifolds).

## What it does

Given a Lagrangian and virtual work, `geomech` automatically derives
globally valid, singularity-free equations of motion on smooth manifolds
(SO(3), S2) using Hamilton's principle and variational calculus --
no Euler angles or quaternions required.

The pipeline:

1. Define symbolic scalars, vectors, and matrices
2. Build kinetic/potential energy and virtual work expressions
3. Call `compute_eom()` -- the library takes variations, applies the
   product rule, expands, integrates by parts, simplifies, and
   extracts the equations

## Installation

Requires Python >= 3.10.

```bash
pip install -e .
```

## Quick start

**Point mass** -- derives `m * ddot_x = f - m*g*e3`:

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

**Spherical pendulum** on S2:

```python
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

**Rigid body** on SO(3):

```python
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

More examples in [`examples/`](examples/).

## Expression types

| Type     | Description                 | Constructor                      |
| -------- | --------------------------- | -------------------------------- |
| `Scalar` | Symbolic scalar             | `Scalar('m', attr=['Constant'])` |
| `Vector` | Column vector (default 3x1) | `Vector('x')`                    |
| `Matrix` | Square matrix (default 3x3) | `Matrix('J', attr=['Constant'])` |
| `S2`     | Unit vector on the 2-sphere | `S2('q')`                        |
| `SO3`    | Rotation matrix on SO(3)    | `SO3('R')`                       |

Batch constructors: `getScalars('a b c')`, `getVectors(['x', 'y'])`, `getMatrices('M N')`.

## Operations

**Arithmetic** -- natural Python operators build expression trees:

```python
a * b          # scalar multiplication
a * x          # scalar-vector multiplication
M * x          # matrix-vector multiplication
x + y          # vector addition
Dot(x, y)      # dot product
Cross(x, y)    # cross product
Hat(x)         # hat map (R^3 -> so(3))
Vee(M)         # vee map (so(3) -> R^3)
Transpose(x)   # transpose
```

**Calculus**:

```python
x.t_diff()       # time derivative
x.t_integrate()  # time integral
x.delta()        # variation (delta)
```

**Transformations**:

```python
expand(expr)        # distribute products over sums
simplify(expr)      # pull scalars, apply identities, eliminate zeros
full_simplify(expr) # expand + simplify until convergence
```

## Package structure

```
geomech/
    core/
        base/           # Expr types, flags, scalars/vectors/matrices, manifolds
        operations/     # Add, Mul, Dot, Cross, Hat, Variation, TimeDerivative, ...
        transformations/# expand, simplify, pull, vector_rules
        math/           # extract_coeff, collect, integrate_by_parts
    dynamics/           # compute_eom, SystemVariables
    utils/              # print_eom, print_tree, render_eom
```

## Running tests

```bash
pip install -e ".[dev]"
pytest
```

## Citation

If you use geomech in your research, please cite along with corresponding original work, [Global Formulations of Lagrangian and Hamiltonian Dynamics on Manifolds](https://link.springer.com/book/10.1007/978-3-319-56953-6) and [Kotaru & Sreenath, 2020](https://link.springer.com/chapter/10.1007/978-3-030-43089-4_22).

```bibtex
@software{kotaru2026geomech,
  title={geomech: Symbolic toolbox for geometric mechanics on smooth manifolds},
  author={Kotaru, Prasanth},
  url={https://github.com/vkotaru/pygeomech},
  year={2026}
}
```

## License

BSD 3-Clause. See [LICENSE](LICENSE).
