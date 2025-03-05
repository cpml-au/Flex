from .primitives import add_primitives_to_pset
from importlib import import_module


def add_primitives_to_pset_from_dict(pset, primitives_dict):
    primitives_collection = dict()
    imports = primitives_dict["imports"].items()

    for module_name, function_names in imports:
        module = import_module(module_name)
        for function_name in function_names:
            primitive = getattr(module, function_name)
            primitives_collection = primitives_collection | primitive

    add_primitives_to_pset(
        pset,
        primitives_dict["used"],
        primitives_collection,
    )

    return pset


def detect_nested_trigonometric_functions(equation):
    # List of trigonometric functions
    trig_functions = ["sin", "cos"]
    nested = 0  # Flag to indicate if nested functions are found
    function_depth = 0  # Track depth within trigonometric function calls
    i = 0

    while i < len(equation) and not nested:
        # Look for trigonometric function
        trig_found = any(
            equation[i : i + len(trig)].lower() == trig for trig in trig_functions
        )
        if trig_found:
            # If a trig function is found, look for its opening parenthesis
            j = i
            while j < len(equation) and equation[j] not in ["(", " "]:
                j += 1
            if j < len(equation) and equation[j] == "(":
                if function_depth > 0:
                    # We are already inside a trig function, this is a nested trig
                    # function
                    nested = 1
                function_depth += 1
                i = j  # Move i to the position of '('
        elif equation[i] == "(" and function_depth > 0:
            # Increase depth if we're already in a trig function
            function_depth += 1
        elif equation[i] == ")":
            if function_depth > 0:
                # Leaving a trigonometric function or nested parentheses
                function_depth -= 1
        i += 1

    return nested
