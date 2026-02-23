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

    # Load the primitives (e.g., coch_primitives)
    for module_name, function_names in imports:
        module = import_module(module_name)
        for function_name in function_names:
            primitive = getattr(module, function_name)
            primitives_collection = primitives_collection | primitive

    for entry in primitives_dict["used"]:
        # Normalize attributes. In YAML they might be None,
        # but in your dictionary they are now lists of Enums.
        dims = entry["dimension"] if entry["dimension"] is not None else []
        ranks = entry["rank"] if entry["rank"] is not None else []

        # Build suffixes using .value.
        # Example: Dimension.ZERO (0) + Rank.VECTOR ("V") -> "0V"
        # We use str() for dimension because IntEnum.value is an int.
        feasible_suffixes = {
            f"{d.value}{r.value}" for d, r in itertools.product(dims, ranks)
        }

        for typed_name, params in primitives_collection.items():
            base_name = get_base_name(typed_name)

            if entry["name"] == base_name:
                # Extract the suffix from the generated name.
                # Example: base="AddC", typed="AddCP0V" -> suffix_info="0V"
                # The [1:] skips the Complex ('P' or 'D')
                suffix_info = typed_name.replace(base_name, "")[1:]

                # 4. Filter and add to pset
                # If feasible_suffixes is empty, it means we don't filter
                # (add all variants)
                if not feasible_suffixes or suffix_info in feasible_suffixes:
                    pset.addPrimitive(
                        params.op, params.in_types, params.out_type, name=typed_name
                    )

    return pset
