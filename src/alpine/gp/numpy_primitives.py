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
    "sqrt": PrimitiveParams(np.sqrt, [float], float),
    "square": PrimitiveParams(np.square, [float], float),
    "aq": PrimitiveParams(
        lambda x, y: np.divide(x, np.sqrt(1 + y**2)), [float, float], float
    ),
}
