# Constant fitting with numba

The function `fit_mse` of the module `flex.gp.numba_eval`, which `fitness_batch` calls
for each individual, fits the constants of a tree (the terminals named `c`) by
minimizing the mean squared error with the Levenberg-Marquardt method. The method uses
the exact derivatives of the tree with respect to its constants, which are computed in
forward mode by the module `flex.gp.numba_grad`. The fit starts from random values,
uniformly distributed within `bounds`, whose generator is seeded with a hash of the
tree, so a given tree always starts from the same values, in serial and in parallel
runs. If a derivative is not finite at some data point, the fit stops and returns the
constants reached so far.

The derivatives at points where the differentiation rules are not defined follow these
conventions:

- In the chain rule, a product with a zero factor is zero, also if the other factor is
  infinite or undefined. This gives the true derivative in cases such as
  `sqrt(mul(c, x))` at `x = 0`, where the tree is 0 for every `c`, and `exp(div(c, x))`
  at `x = 0` with `c < 0`, where the tree underflows to 0. The convention is not always
  exact:
  - where the true derivative does not exist but the inner derivative is zero, as for
    `sqrt(square(sub(x, c)))`, which is `|x - c|`, at `x = c`, the derivative is 0 and the
    fit continues instead of stopping;
  - where the limit of the product is not zero, the derivative is wrong. For example,
    `div(x, div(x, sub(c, c)))` is `c1 - c2`, but with two equal constants its derivative
    is computed as 0 instead of 1. This requires a denominator that depends on the
    constants to be exactly zero.
- At a zero base, `pow(a, b)` with `b >= 1` uses the derivative from the right,
  `b * 0^(b - 1)`, also when `b` is not an integer and `a^b` is not a real number for
  negative `a`. For example, `pow(sub(x, c), 1.5)` at `x = c` has a finite derivative
  and the fit continues, even if the tree is not finite for larger `c`.

A step of the fit is accepted only if it reduces the error, which is computed from the
values of the tree: an inaccurate derivative can make the fit less effective, but it does
not affect the reported error, and steps to constants at which the tree is not finite on
the data are rejected.

A tree that is not finite on the data at the start values is not fitted, also if a limit
exists at the point, as for `div(sin(mul(c, x)), x)` at `x = 0`; since the start values
depend only on the tree, this happens at each evaluation of that tree. Finally, `sin` and
`cos` of very large values, such as `sin(exp(mul(c, x)))` at `c * x = 40`, are not
accurate in double precision: the value and the derivative of the tree are finite but
unreliable.
