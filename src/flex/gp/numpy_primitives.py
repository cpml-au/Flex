from .primitives import PrimitiveParams
import numpy as np


numpy_primitives = {
    "add": PrimitiveParams(np.add, [float, float], float),
    "sub": PrimitiveParams(np.subtract, [float, float], float),
    "mul": PrimitiveParams(np.multiply, [float, float], float),
    "div": PrimitiveParams(np.divide, [float, float], float),
    "sin": PrimitiveParams(np.sin, [float], float),
    "arcsin": PrimitiveParams(np.arcsin, [float], float),
    "cos": PrimitiveParams(np.cos, [float], float),
    "arccos": PrimitiveParams(np.arccos, [float], float),
    "exp": PrimitiveParams(np.exp, [float], float),
    "log": PrimitiveParams(np.log, [float], float),
    "prot_log": PrimitiveParams(lambda x: np.log(np.abs(x)), [float], float),
    "pow": PrimitiveParams(np.pow, [float, float], float),
    "prot_pow": PrimitiveParams(
        lambda x, k: np.pow(np.abs(x), k), [float, float], float
    ),
    "sqrt": PrimitiveParams(np.sqrt, [float], float),
    "square": PrimitiveParams(np.square, [float], float),
    "aq": PrimitiveParams(
        lambda x, y: np.divide(x, np.sqrt(1 + y**2)), [float, float], float
    ),
    "tanh": PrimitiveParams(np.tanh, [float], float),
}

conversion_rules = {
    "sub": lambda *args_: "Add({}, Mul(-1,{}))".format(*args_),
    "div": lambda *args_: "Mul({}, Pow({}, -1))".format(*args_),
    "mul": lambda *args_: "Mul({},{})".format(*args_),
    "add": lambda *args_: "Add({},{})".format(*args_),
    "pow": lambda *args_: "Pow({}, {})".format(*args_),
    "square": lambda *args_: "Pow({}, 2)".format(*args_),
    "aq": lambda *args_: "Mul({}, Pow(Add(1, Pow({}, 2), -1))".format(*args_),
    "prot_pow": lambda *args_: "Pow(Abs({}), {})".format(*args_),
    "prot_log": lambda *args_: "log(Abs({}))".format(*args_),
}
