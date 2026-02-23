from dctkit.dec import cochain as C
from functools import partial
import jax.numpy
from typing import Callable, List, Dict
import itertools
from .primitives import PrimitiveParams
from importlib import import_module
from enum import Enum, IntEnum

# Define the modules and functions needed to eval inputs and outputs
modules_functions = {"dctkit.dec": ["cochain"]}


class Complex(Enum):
    """Enum class for complex."""

    PRIMAL = "P"
    DUAL = "D"


class Rank(Enum):
    """Enum class for rank."""

    SCALAR = ""
    VECTOR = "V"
    TENSOR = "T"


class Dimension(IntEnum):
    """Enum class for dimension."""

    ZERO = 0
    ONE = 1
    TWO = 2


class CochainBasePrimitive:
    """A simple class to handle a cochain base primitive function.

    Args:
        base_name: name of the base primitive.
        base_fun: callable base function.
        input: a list containing the input types (str) of `base_fun`.
        output: a string containing the output type of `base_fun`.
        att_input: a dictionary with keys 'complex' (primal/dual), 'dimension' (0,1,2),
            and 'rank' ("SC", i.e. scalar, "V", "T").
        map_rule: a dictionary consisting of the same keys of `att_input`. In this case,
            each key contains a callable object that provides the map to get the output
            complex/dimension/rank given the input one.
    """

    def __init__(
        self,
        base_name: str,
        base_fun: Callable,
        input: List[str],
        output: str,
        att_input: Dict,
        map_rule: Dict,
    ):
        self.base_name = base_name
        self.base_fun = base_fun
        self.input = input
        self.output = output
        self.att_input = att_input
        self.map_rule = map_rule


def inv_scalar_mul(c: C.Cochain, f: float):
    """Scalar multiplication between a cochain and the inverse of a float.

    Args:
        c: a cochain.
        f: a float.

    Returns:
        the scalar product between c and 1/f.
    """
    try:
        return C.scalar_mul(c, 1 / f)
    except ZeroDivisionError:
        return C.scalar_mul(c, jax.numpy.nan)


def switch_complex(complex: Complex):
    """Switch complex (from primal to dual or viceversa).
    Args:
        complex: a complex object.

    Returns:
        the other complex.
    """
    # Logic to toggle between P and D
    return Complex.DUAL if complex == Complex.PRIMAL else Complex.PRIMAL


def define_eval_with_suitable_imports(imports: Dict):
    """Creates a scoped evaluation function with pre-loaded modules.

    This prevents repetitive imports and ensures that string-based type
    definitions (like "IntP1V") can be converted into actual class
    references during GP tree construction.

    Args:
        imports: A dictionary where keys are module paths and values
            are lists of function/class names to import.

    Returns:
        A function `eval_with_globals(expression)` that evaluates strings
        within the context of the imported components.
    """
    custom_globals = {}
    for module_name, function_names in imports.items():
        module = import_module(module_name)
        for function_name in function_names:
            custom_globals[function_name] = getattr(module, function_name)

    def eval_with_globals(expression):
        return eval(expression, custom_globals)

    return eval_with_globals


def compute_primitive_in_out_type(
    primitive: CochainBasePrimitive,
    eval_with_globals: Callable,
    in_complex: Complex,
    in_dim: Dimension,
    in_rank: Rank,
):
    """Resolves the specific variant name and types for a primitive.

    Based on the input complex (Primal/Dual), dimension (0, 1, 2), and
    rank (Scalar, Vector, Tensor), this function generates a unique name
    for the primitive variant, resolves the Python types for all input
    arguments, and calculates the resulting output type using defined
    mapping rules.

    Args:
        primitive: a `CochainBasePrimitive` object.
        eval_with_globals: The evaluation function created by
            `define_eval_with_suitable_imports`.
        in_complex: The current complex.
        in_dim: The current dimension.
        in_rank: The current rank.

    Returns:
        A tuple containing the concatenated name (e.g., "addP1V"),
            a list of resolved Python type objects for inputs and the
            resolved Python type object for the output.
    """
    # # compute the primitive name taking into account
    # # the right complex, dim and rank
    # base_primitive = primitive["fun_info"]
    map_rule = primitive.map_rule

    # in_rank = in_rank.replace("SC", "")
    primitive_name = (
        f"{primitive.base_name}{in_complex.value}{in_dim.value}{in_rank.value}"
    )
    in_type_name = []
    # compute the input type list
    for i, input in enumerate(primitive.input):
        # float type must be handled separately
        if input == "float":
            in_type_name.append(input)
        else:
            in_type_name.append(
                f"{input}{in_complex.value}{in_dim.value}{in_rank.value}"
            )
    in_type = list(map(eval_with_globals, in_type_name))

    # 1. Run the mapping rules
    raw_out_complex = map_rule["complex"](in_complex)
    raw_out_dim = map_rule["dimension"](in_dim)
    raw_out_rank = map_rule["rank"](in_rank)

    # 2. Safely cast back to Enums ONLY if they aren't None
    out_complex = Complex(raw_out_complex) if raw_out_complex is not None else None
    out_dim = Dimension(raw_out_dim) if raw_out_dim is not None else None
    out_rank = Rank(raw_out_rank) if raw_out_rank is not None else None

    # 3. Build the output string name
    # We filter out None values so "float" + None + None becomes just "float"
    out_type_parts = [primitive.output]
    for attr in [out_complex, out_dim, out_rank]:
        if attr is not None:
            # NOTE: use str(attr.value) for Dimension to get "0"
            # instead of "<Dimension.ZERO: 0>"
            out_type_parts.append(str(attr.value))

    out_type_name = "".join(out_type_parts)
    out_type = eval_with_globals(out_type_name)

    return primitive_name, in_type, out_type


def generate_primitive_variants(primitive: CochainBasePrimitive, imports: Dict):
    """Generate primitive variants given a typed primitive.

    Args:
        primitive: a `CochainBasePrimitive` object.
        imports: dictionary whose keys and values are the modules and the functions to
            be imported in order to evaluate the input/output types of the primitive.

    Returns:
        a dict in which each key is the name of the primitive variant and each value
            is a PrimitiveParams object.
    """
    in_attribute = primitive.att_input
    primitive_dictionary = dict()

    # Dynamically import modules and functions needed to eval input/output types
    eval_with_globals = define_eval_with_suitable_imports(imports)

    # Create an iterator for all combinations
    combinations = itertools.product(
        in_attribute["complex"], in_attribute["dimension"], in_attribute["rank"]
    )

    for in_cat, in_dim, in_rank in combinations:
        primitive_name, in_type, out_type = compute_primitive_in_out_type(
            primitive, eval_with_globals, in_cat, in_dim, in_rank
        )
        primitive_dictionary[primitive_name] = PrimitiveParams(
            primitive.base_fun, in_type, out_type
        )
    return primitive_dictionary


# define cochain primitives
add_coch = CochainBasePrimitive(
    base_name="AddC",
    base_fun=C.add,
    input=["cochain.Cochain", "cochain.Cochain"],
    output="cochain.Cochain",
    att_input={
        "complex": (Complex.PRIMAL, Complex.DUAL),
        "dimension": (Dimension.ZERO, Dimension.ONE, Dimension.TWO),
        "rank": (Rank.SCALAR, Rank.VECTOR, Rank.TENSOR),
    },
    map_rule={
        "complex": lambda x: x,
        "dimension": lambda x: x,
        "rank": lambda x: x,
    },
)
sub_coch = CochainBasePrimitive(
    base_name="SubC",
    base_fun=C.sub,
    input=["cochain.Cochain", "cochain.Cochain"],
    output="cochain.Cochain",
    att_input={
        "complex": (Complex.PRIMAL, Complex.DUAL),
        "dimension": (Dimension.ZERO, Dimension.ONE, Dimension.TWO),
        "rank": (Rank.SCALAR, Rank.VECTOR, Rank.TENSOR),
    },
    map_rule={
        "complex": lambda x: x,
        "dimension": lambda x: x,
        "rank": lambda x: x,
    },
)
coboundary = CochainBasePrimitive(
    base_name="cob",
    base_fun=C.coboundary,
    input=["cochain.Cochain"],
    output="cochain.Cochain",
    att_input={
        "complex": (Complex.PRIMAL, Complex.DUAL),
        "dimension": (Dimension.ZERO, Dimension.ONE),
        "rank": (Rank.SCALAR, Rank.VECTOR, Rank.TENSOR),
    },
    map_rule={
        "complex": lambda x: x,
        "dimension": lambda x: Dimension(x + 1),
        "rank": lambda x: x,
    },
)
codifferential = CochainBasePrimitive(
    base_name="del",
    base_fun=C.codifferential,
    input=["cochain.Cochain"],
    output="cochain.Cochain",
    att_input={
        "complex": (Complex.PRIMAL, Complex.DUAL),
        "dimension": (Dimension.ONE, Dimension.TWO),
        "rank": (Rank.SCALAR, Rank.VECTOR, Rank.TENSOR),
    },
    map_rule={
        "complex": lambda x: x,
        "dimension": lambda x: Dimension(x - 1),
        "rank": lambda x: x,
    },
)
tr_coch = CochainBasePrimitive(
    base_name="tr",
    base_fun=C.trace,
    input=["cochain.Cochain"],
    output="cochain.Cochain",
    att_input={
        "complex": (Complex.PRIMAL, Complex.DUAL),
        "dimension": (Dimension.ZERO, Dimension.ONE, Dimension.TWO),
        "rank": (Rank.TENSOR,),
    },
    map_rule={
        "complex": lambda x: x,
        "dimension": lambda x: x,
        "rank": lambda x: x,
    },
)
mul_FT = CochainBasePrimitive(
    base_name="MF",
    base_fun=C.scalar_mul,
    input=["cochain.Cochain", "float"],
    output="cochain.Cochain",
    att_input={
        "complex": (Complex.PRIMAL, Complex.DUAL),
        "dimension": (Dimension.ZERO, Dimension.ONE, Dimension.TWO),
        "rank": (Rank.SCALAR, Rank.VECTOR, Rank.TENSOR),
    },
    map_rule={
        "complex": lambda x: x,
        "dimension": lambda x: x,
        "rank": lambda x: x,
    },
)
inv_mul_FT = CochainBasePrimitive(
    base_name="InvM",
    base_fun=inv_scalar_mul,
    input=["cochain.Cochain", "float"],
    output="cochain.Cochain",
    att_input={
        "complex": (Complex.PRIMAL, Complex.DUAL),
        "dimension": (Dimension.ZERO, Dimension.ONE, Dimension.TWO),
        "rank": (Rank.SCALAR, Rank.VECTOR, Rank.TENSOR),
    },
    map_rule={
        "complex": lambda x: x,
        "dimension": lambda x: x,
        "rank": lambda x: x,
    },
)
mul_coch = CochainBasePrimitive(
    base_name="CMul",
    base_fun=C.cochain_mul,
    input=["cochain.Cochain", "cochain.Cochain"],
    output="cochain.Cochain",
    att_input={
        "complex": (Complex.PRIMAL, Complex.DUAL),
        "dimension": (Dimension.ZERO, Dimension.ONE, Dimension.TWO),
        "rank": (Rank.SCALAR,),
    },
    map_rule={
        "complex": lambda x: x,
        "dimension": lambda x: x,
        "rank": lambda x: x,
    },
)
tran_coch = CochainBasePrimitive(
    base_name="tran",
    base_fun=C.transpose,
    input=["cochain.Cochain"],
    output="cochain.Cochain",
    att_input={
        "complex": (Complex.PRIMAL, Complex.DUAL),
        "dimension": (Dimension.ZERO, Dimension.ONE, Dimension.TWO),
        "rank": (Rank.TENSOR,),
    },
    map_rule={
        "complex": lambda x: x,
        "dimension": lambda x: x,
        "rank": lambda x: x,
    },
)
sym_coch = CochainBasePrimitive(
    base_name="sym",
    base_fun=C.sym,
    input=["cochain.Cochain"],
    output="cochain.Cochain",
    att_input={
        "complex": (Complex.PRIMAL, Complex.DUAL),
        "dimension": (Dimension.ZERO, Dimension.ONE, Dimension.TWO),
        "rank": (Rank.TENSOR,),
    },
    map_rule={
        "complex": lambda x: x,
        "dimension": lambda x: x,
        "rank": lambda x: x,
    },
)
star_1 = CochainBasePrimitive(
    base_name="St1",
    base_fun=C.star,
    input=["cochain.Cochain"],
    output="cochain.Cochain",
    att_input={
        "complex": (Complex.PRIMAL, Complex.DUAL),
        "dimension": (Dimension.ZERO, Dimension.ONE),
        "rank": (Rank.SCALAR, Rank.VECTOR, Rank.TENSOR),
    },
    map_rule={
        "complex": switch_complex,
        "dimension": lambda x: Dimension(1 - x),
        "rank": lambda x: x,
    },
)
star_2 = CochainBasePrimitive(
    base_name="St2",
    base_fun=C.star,
    input=["cochain.Cochain"],
    output="cochain.Cochain",
    att_input={
        "complex": (Complex.PRIMAL, Complex.DUAL),
        "dimension": (Dimension.ZERO, Dimension.ONE, Dimension.TWO),
        "rank": (Rank.SCALAR, Rank.VECTOR, Rank.TENSOR),
    },
    map_rule={
        "complex": switch_complex,
        "dimension": lambda x: Dimension(2 - x),
        "rank": lambda x: x,
    },
)
inner_product = CochainBasePrimitive(
    base_name="Inn",
    base_fun=C.inner,
    input=["cochain.Cochain", "cochain.Cochain"],
    output="float",
    att_input={
        "complex": (Complex.PRIMAL, Complex.DUAL),
        "dimension": (Dimension.ZERO, Dimension.ONE, Dimension.TWO),
        "rank": (Rank.SCALAR, Rank.VECTOR, Rank.TENSOR),
    },
    map_rule={
        "complex": lambda x: None,
        "dimension": lambda x: None,
        "rank": lambda x: None,
    },
)
sin_coch = CochainBasePrimitive(
    base_name="Sin",
    base_fun=C.sin,
    input=["cochain.Cochain"],
    output="cochain.Cochain",
    att_input={
        "complex": (Complex.PRIMAL, Complex.DUAL),
        "dimension": (Dimension.ZERO, Dimension.ONE, Dimension.TWO),
        "rank": (Rank.SCALAR, Rank.VECTOR, Rank.TENSOR),
    },
    map_rule={
        "complex": lambda x: x,
        "dimension": lambda x: x,
        "rank": lambda x: x,
    },
)
arcsin_coch = CochainBasePrimitive(
    base_name="ArcSin",
    base_fun=C.arcsin,
    input=["cochain.Cochain"],
    output="cochain.Cochain",
    att_input={
        "complex": (Complex.PRIMAL, Complex.DUAL),
        "dimension": (Dimension.ZERO, Dimension.ONE, Dimension.TWO),
        "rank": (Rank.SCALAR, Rank.VECTOR, Rank.TENSOR),
    },
    map_rule={
        "complex": lambda x: x,
        "dimension": lambda x: x,
        "rank": lambda x: x,
    },
)
cos_coch = CochainBasePrimitive(
    base_name="Cos",
    base_fun=C.cos,
    input=["cochain.Cochain"],
    output="cochain.Cochain",
    att_input={
        "complex": (Complex.PRIMAL, Complex.DUAL),
        "dimension": (Dimension.ZERO, Dimension.ONE, Dimension.TWO),
        "rank": (Rank.SCALAR, Rank.VECTOR, Rank.TENSOR),
    },
    map_rule={
        "complex": lambda x: x,
        "dimension": lambda x: x,
        "rank": lambda x: x,
    },
)
arccos_coch = CochainBasePrimitive(
    base_name="ArcCos",
    base_fun=C.arccos,
    input=["cochain.Cochain"],
    output="cochain.Cochain",
    att_input={
        "complex": (Complex.PRIMAL, Complex.DUAL),
        "dimension": (Dimension.ZERO, Dimension.ONE, Dimension.TWO),
        "rank": (Rank.SCALAR, Rank.VECTOR, Rank.TENSOR),
    },
    map_rule={
        "complex": lambda x: x,
        "dimension": lambda x: x,
        "rank": lambda x: x,
    },
)
exp_coch = CochainBasePrimitive(
    base_name="Exp",
    base_fun=C.exp,
    input=["cochain.Cochain"],
    output="cochain.Cochain",
    att_input={
        "complex": (Complex.PRIMAL, Complex.DUAL),
        "dimension": (Dimension.ZERO, Dimension.ONE, Dimension.TWO),
        "rank": (Rank.SCALAR, Rank.VECTOR, Rank.TENSOR),
    },
    map_rule={
        "complex": lambda x: x,
        "dimension": lambda x: x,
        "rank": lambda x: x,
    },
)
log_coch = CochainBasePrimitive(
    base_name="Log",
    base_fun=C.log,
    input=["cochain.Cochain"],
    output="cochain.Cochain",
    att_input={
        "complex": (Complex.PRIMAL, Complex.DUAL),
        "dimension": (Dimension.ZERO, Dimension.ONE, Dimension.TWO),
        "rank": (Rank.SCALAR, Rank.VECTOR, Rank.TENSOR),
    },
    map_rule={
        "complex": lambda x: x,
        "dimension": lambda x: x,
        "rank": lambda x: x,
    },
)
sqrt_coch = CochainBasePrimitive(
    base_name="Sqrt",
    base_fun=C.sqrt,
    input=["cochain.Cochain"],
    output="cochain.Cochain",
    att_input={
        "complex": (Complex.PRIMAL, Complex.DUAL),
        "dimension": (Dimension.ZERO, Dimension.ONE, Dimension.TWO),
        "rank": (Rank.SCALAR, Rank.VECTOR, Rank.TENSOR),
    },
    map_rule={
        "complex": lambda x: x,
        "dimension": lambda x: x,
        "rank": lambda x: x,
    },
)
square_coch = CochainBasePrimitive(
    base_name="Square",
    base_fun=C.square,
    input=["cochain.Cochain"],
    output="cochain.Cochain",
    att_input={
        "complex": (Complex.PRIMAL, Complex.DUAL),
        "dimension": (Dimension.ZERO, Dimension.ONE, Dimension.TWO),
        "rank": (Rank.SCALAR, Rank.VECTOR, Rank.TENSOR),
    },
    map_rule={
        "complex": lambda x: x,
        "dimension": lambda x: x,
        "rank": lambda x: x,
    },
)

coch_prim_list = [
    add_coch,
    sub_coch,
    coboundary,
    codifferential,
    tr_coch,
    mul_FT,
    inv_mul_FT,
    mul_coch,
    tran_coch,
    sym_coch,
    star_1,
    star_2,
    inner_product,
    sin_coch,
    arcsin_coch,
    cos_coch,
    arccos_coch,
    exp_coch,
    log_coch,
    sqrt_coch,
    square_coch,
]

coch_primitives = list(
    map(partial(generate_primitive_variants, imports=modules_functions), coch_prim_list)
)
coch_primitives = {k: v for d in coch_primitives for k, v in d.items()}
