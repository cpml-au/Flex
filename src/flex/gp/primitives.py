from deap.gp import PrimitiveSetTyped
from typing import List, Dict, Callable, Type
from importlib import import_module
import itertools


class PrimitiveParams:
    """A simple class to handle a primitive function.

    Args:
        op: the callable function.
        in_types: input types of the primitive.
        out_type: output type of the primitive.
    """

    def __init__(self, op: Callable, in_types: Type, out_type: Type):
        self.op = op
        self.in_types = in_types
        self.out_type = out_type


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
    primitive, eval_with_globals, in_complex, in_dim, in_rank
):
    """Resolves the specific variant name and types for a primitive.

    Based on the input complex (Primal/Dual), dimension (0, 1, 2), and
    rank (Scalar, Vector, Tensor), this function generates a unique name
    for the primitive variant, resolves the Python types for all input
    arguments, and calculates the resulting output type using defined
    mapping rules.

    Args:
        primitive: the base primitive configuration dictionary, see the
            documentation of `generate_primitive_variants`
        eval_with_globals: The evaluation function created by
            `define_eval_with_suitable_imports`.
        in_complex: The current complex ('P' or 'D').
        in_dim: The current dimension as a string.
        in_rank: The current rank (e.g., 'SC', 'V', 'T').

    Returns:
        A tuple containing the concatenated name (e.g., "addP1V"),
            a list of resolved Python type objects for inputs and the
            resolved Python type object for the output.
    """
    # # compute the primitive name taking into account
    # # the right complex, dim and rank
    base_primitive = primitive["fun_info"]
    map_rule = primitive["map_rule"]

    in_rank = in_rank.replace("SC", "")
    primitive_name = base_primitive["name"] + in_complex + in_dim + in_rank
    in_type_name = []
    # compute the input type list
    for i, input in enumerate(primitive["input"]):
        # float type must be handled separately
        if input == "float":
            in_type_name.append(input)
        elif len(in_rank) == 2:
            # in this case the correct rank must be taken
            in_type_name.append(input + in_complex + in_dim + in_rank[i])
        else:
            in_type_name.append(input + in_complex + in_dim + in_rank)
    in_type = list(map(eval_with_globals, in_type_name))
    out_complex = map_rule["complex"](in_complex)
    out_dim = str(map_rule["dimension"](int(in_dim)))
    out_rank = map_rule["rank"](in_rank)
    out_type_name = primitive["output"] + out_complex + out_dim + out_rank
    out_type = eval_with_globals(out_type_name)
    return primitive_name, in_type, out_type


def generate_primitive_variants(
    primitive: Dict[str, Dict[str, Callable] | List[str] | str | Dict], imports: Dict
):
    """Generate primitive variants given a typed primitive.

    Args:
        primitive: a dictionary containing the relevant information of the function.
          It consists of the following 5 keys: 'fun_info' contains an inner dictionary
          encoding the name of the function (value of the inner key 'name') and the
          callable itself (value of the inner key 'fun'); 'input' contains a list
          composed of the input types; 'output' contains a string encoding the output
          type; 'att_input' contains an inner dictionary with keys 'complex'
          (primal/dual), 'dimension' (0,1,2) and 'rank' ("SC", i.e. scalar, "V", "T"
          or "VT"); 'map_output' contains an inner dictionary consisting of the
          same keys of 'att_input'. In this case, each key contains a callable object
          that provides the map to get the output complex/dimension/rank given the
          input one.
        imports: dictionary whose keys and values are the modules and the functions to
            be imported in order to evaluate the input/output types of the primitive.

    Returns:
        a dict in which each key is the name of the primitive variant and each value
            is a PrimitiveParams object.
    """
    base_primitive = primitive["fun_info"]
    in_attribute = primitive["att_input"]
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
            base_primitive["fun"], in_type, out_type
        )
    return primitive_dictionary


def get_base_name(typed_name: str):
    """Extracts the base name by removing P/D and rank/dim indicators.

    Args:
        typed_name: the full name of the primitive (e.g., "St1D1V").

    Returns:
        the base name of the primitive (e.g., St1).
    """
    replacements = ["P", "D", "0", "1", "2", "V", "T"]
    # Preserve the first character, clean the rest of the suffix
    suffix_part = typed_name[1:]
    for r in replacements:
        suffix_part = suffix_part.replace(r, "")
    return typed_name[0] + suffix_part


def add_primitives_to_pset_from_dict(pset: PrimitiveSetTyped, primitives_dict: Dict):
    """Add a given set of primitives to a PrimitiveSetTyped object.

    Args:
        pset: a primitive set.
        primitives_dict: a dictionary composed of two keys: `imports`, containing the
          import location of the pre-defined primitives; `used`, containing a list of
          dictionaries (of the same structure as the one in `add_primitives_to_pset`).

    Returns:
        the updated primitive set
    """
    primitives_collection = dict()
    imports = primitives_dict["imports"].items()

    for module_name, function_names in imports:
        module = import_module(module_name)
        for function_name in function_names:
            primitive = getattr(module, function_name)
            primitives_collection = primitives_collection | primitive

    for entry in primitives_dict["used"]:
        # Normalize "None" strings to empty lists
        dims = entry["dimension"] if entry["dimension"] != "None" else []
        ranks = entry["rank"] if entry["rank"] != "None" else []

        # Build suffixes: e.g., '0' + 'V' -> '0V'
        feasible_suffixes = {
            f"{d}{r.replace('SC', '')}" for d, r in itertools.product(dims, ranks)
        }

        for typed_name, params in primitives_collection.items():
            true_name = get_base_name(typed_name)

            if entry["name"] == true_name:
                # Get the part after the true name (e.g., 'P0V' -> 'P0V')
                # Then skip the first char (Primal/Dual) to get the dim/rank suffix
                suffix_info = typed_name.replace(true_name, "")[1:]

                if not feasible_suffixes or suffix_info in feasible_suffixes:
                    pset.addPrimitive(
                        params.op, params.in_types, params.out_type, name=typed_name
                    )

    return pset
