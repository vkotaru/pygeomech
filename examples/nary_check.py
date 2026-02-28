from geomech import *

add = Add()
a, b, c, d, e = getScalars('a b c d e')
f, g = getScalars('f g', attr=['Constant'])
a0 = Scalar('0', value=0)
a1 = Scalar('1', value=1)
a2 = Scalar('2', value=2)

sum_scalars_ = Add(a, b, Add(c, d, e), f + g, [a1, a2])
sum_scalars = simplify(sum_scalars_)
delta_sum_scalars = sum_scalars.delta()
dot_sum_scalars = sum_scalars.t_diff()
dot_sum2_ = Add(f, g).t_diff()
scalar_scalar_mul = a * sum_scalars
# sum_scalars2 = sum_scalars.copy()
# sum_scalars2.replace_at(0, e)

v, u = getVectors('u v', attr=['Constant'])
x, y, z = getVectors(['x', 'y', 'z'])
sum_vectors = VAdd(u, v)
sum_vectors += VAdd(x, y, z)

A, B, C = getMatrices('A B C')
sum_matrices = A + MAdd(B, C)
dot_sum_matrices = sum_matrices.t_diff()

print('done')
