from .primitives import PrimitiveParams
import jax.numpy as jnp
import dctkit as dt


def protectedDiv(left, right):
    try:
        return jnp.divide(left, right)
    except ZeroDivisionError:
        return jnp.nan


def protectedLog(x):
    try:
        return jnp.log(x)
    except ValueError:
        return jnp.nan


def protectedSqrt(x):
    try:
        return jnp.sqrt(x)
    except ValueError:
        return jnp.nan


def inv_float(x):
    return protectedDiv(1.0, x)


def square_mod(x):
    return jnp.square(x).astype(dt.float_dtype)


jax_primitives = {
    # scalar operations (JAX backend)
    "AddF": PrimitiveParams(jnp.add, [float, float], float),
    "SubF": PrimitiveParams(jnp.subtract, [float, float], float),
    "MulF": PrimitiveParams(jnp.multiply, [float, float], float),
    "Div": PrimitiveParams(protectedDiv, [float, float], float),
    "SinF": PrimitiveParams(jnp.sin, [float], float),
    "ArcsinF": PrimitiveParams(jnp.arcsin, [float], float),
    "CosF": PrimitiveParams(jnp.cos, [float], float),
    "ArccosF": PrimitiveParams(jnp.arccos, [float], float),
    "ExpF": PrimitiveParams(jnp.exp, [float], float),
    "LogF": PrimitiveParams(protectedLog, [float], float),
    "SqrtF": PrimitiveParams(protectedSqrt, [float], float),
    "SquareF": PrimitiveParams(jnp.square, [float], float),
    "InvF": PrimitiveParams(inv_float, [float], float),
}
