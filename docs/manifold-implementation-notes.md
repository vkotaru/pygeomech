# Manifold Implementation Notes

## Status: S2 `delta(omega)` formula needs verification

The current Python `TS2.delta(substitute=True)` uses:
```
delta(omega) = xi_dot - omega x xi
```
This formula is **not confirmed correct** and does not appear in the Scala reference.
User will provide the correct math.

---

## What was done

### 1. SO3 parity with S2
- `SO3.get_variation_vector()` now returns `TSO3` (was plain `Vector`)
- `TSO3.delta(substitute=True)` uses `eta_dot + Omega x eta` (from Scala `delta_Omega()`)
- `manifold_rules.py` extended to handle both S2 and SO3

### 2. S2 simplification rules (vector_rules.py)
- BAC-CAB identity: `cross(q, cross(a, q)) = a` when `||q||=1` and `a` is tangent
- Variants: `cross(q, cross(q, a)) = -a`, `cross(cross(q,a), q) = a`, etc.
- `cross(a, cross(a, q)) = -q * dot(a,a)` when `a` is tangent
- Recursion into Cross children, VAdd, SVMul

### 3. Simplification rules (simplify.py)
- `Hat(v) * w -> Cross(v, w)` in MVMul
- `MMMul(A, B) * v -> A * (B * v)` matrix associativity in MVMul
- `Transpose(Hat(v)) -> -Hat(v)` (skew-symmetry)
- `Transpose(SMMul(M, s)) -> SMMul(Transpose(M), s)`
- `Transpose(MMMul(A, B)) -> MMMul(Transpose(B), Transpose(A))`

### 4. Expand rules (expand.py)
- `Hat(a + b) -> Hat(a) + Hat(b)` (linearity)
- `Vee(a + b) -> Vee(a) + Vee(b)` (linearity)

### 5. Collect rules (collect.py)
- `Dot(R*a, R*b) -> Dot(a, b)` for SO3 rotation stripping
- `Dot(Cross(vec, a), b)` — flip and use scalar triple product
- `Dot(M*Cross(vec, b), w)` — deep extraction through MVMul/Cross
- Nested vec inside Cross on right side of Dot

### 6. Tests
- `test_manifold_rules.py` — 17 tests for S2/SO3 manifold substitutions
- `test_manifold_pipelines.py` — 28 tests for end-to-end pipelines
- Previously skipped S2 standard form tests now pass
- SO3 standard form test updated (M now has inertia terms)
- 920 total tests passing

---

## Key finding: Scala vs Python S2 approach

### Scala approach (works at q level)
Three sequential substitutions in `applyManifoldRules`:
```scala
expr.subs(Delta(Dif(q)), dot_delta_q())   // delta(q_dot) -> xi_dot x q + xi x (omega x q)
expr.subs(Dif(q),        dot_q())         // q_dot        -> omega x q
expr.subs(Delta(q),      delta_q())       // delta(q)     -> xi x q
```
Scala **never computes `delta(omega)` explicitly**.

### Python approach (works at omega level)
- `S2.t_diff()` eagerly returns `Cross(omega, q)` — tree never has `TimeDerivative(q)`
- `S2.delta()` eagerly returns `Cross(xi, q)` — tree never has `Variation(q)`
- `manifold_rules` substitutes `Variation(omega)` -> `xi_dot - omega x xi`

The two approaches are mathematically equivalent via the Jacobi identity,
but the Python approach produces more intermediate terms requiring extra
simplification rules (BAC-CAB, skew-symmetry, etc.).

### Open question
The `delta(omega) = xi_dot - omega x xi` formula needs to be verified.
The Scala code avoids this formula entirely. S2 and SO3 are different
manifolds — there is no reason to assume the variation formula follows
the same pattern as SO3's `delta(Omega) = eta_dot + Omega x eta`.

---

## SO3 reference (confirmed from Scala)

From `Matrices.scala:80`:
```scala
def delta_Omega() = MVMul(Hat(Omega), eta) + eta.diff()
// = Hat(Omega)*eta + eta_dot = Omega x eta + eta_dot
```

From `SmoothManifold.scala:21-29` (four substitutions):
```scala
expr.subs(Delta(Dif(R)),        d/dt(delta_R))
expr.subs(Dif(R),               R * Hat(Omega))
expr.subs(Delta(R),             R * Hat(eta))
expr.subs(Delta(Omega),         eta_dot + Omega x eta)
```

---

## Files changed

### Source
- `geomech/core/base/expressions.py` — SO3/TS2/TSO3 types
- `geomech/core/transformations/manifold_rules.py` — S2 + SO3 substitutions
- `geomech/core/transformations/vector_rules.py` — S2 cross product identities
- `geomech/core/transformations/simplify.py` — Hat/Transpose/associativity
- `geomech/core/transformations/expand.py` — Hat/Vee linearity
- `geomech/core/math/collect.py` — SO3 rotation stripping, deeper patterns

### Tests
- `tests/test_manifold_rules.py` — new (17 tests)
- `tests/test_manifold_pipelines.py` — new (28 tests)
- `tests/test_manifolds.py` — updated SO3 variation vector type
- `tests/test_standard_form.py` — unskipped S2, updated SO3
- `tests/test_expand.py` — updated Hat linearity
- `tests/test_expansion.py` — updated Hat linearity
- `tests/test_collect.py` — updated MVMul Transpose pattern
