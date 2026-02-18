from deap.gp import PrimitiveSetTyped
from typing import Dict, Callable, Type
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
            base_name = get_base_name(typed_name)

            if entry["name"] == base_name:
                # Get the part after the true name (e.g., 'P0V' -> 'P0V')
                # Then skip the first char (Primal/Dual) to get the dim/rank suffix
                suffix_info = typed_name.replace(base_name, "")[1:]

                if not feasible_suffixes or suffix_info in feasible_suffixes:
                    pset.addPrimitive(
                        params.op, params.in_types, params.out_type, name=typed_name
                    )

    return pset
