from deap.gp import PrimitiveSetTyped, Primitive, PrimitiveTree
from typing import List, Dict, Callable, Tuple
import importlib


def switch_category(categories: Tuple, category: str):
    switched_category_list = list(set(categories) - set(category))
    return str(switched_category_list[0])


class PrimitiveParams:
    def __init__(self, op, in_types, out_type) -> None:
        self.op = op
        self.in_types = in_types
        self.out_type = out_type


def generate_primitive_variants(
    primitive: Dict[str, Dict[str, Callable] | List[str] | str | Dict],
    imports: Dict = None,
) -> Dict:
    """Generate primitive variants given a typed primitive.

    Args:
        primitive: a dictionary containing the relevant information of the function.
          It consists of the following 5 keys: 'fun_info' contains an inner dictionary
          encoding the name of the function (value of the inner key 'name') and the
          callable itself (value of the inner key 'fun'); 'input' contains a list
          composed of the input types; 'output' contains a string encoding the output
          type; 'att_input' contains an inner dictionary with keys 'category'
          (primal/dual), 'dimension' (0,1,2) and 'rank' ("SC", i.e. scalar, "V", "T"
          or "VT"); 'map_output' contains an inner dictionary consisting of the
          same keys of 'att_input'. In this case, each key contains a callable object
          that provides the map to get the output category/dimension/rank given the
          input one.
        imports: dictionary whose keys and values are the modules and the functions to
            be imported in order to evaluate the input/output types of the primitive.

    Returns:
        a dict in which each key is the name of the primitive variant and each value
            is a PrimitiveParams object.
    """
    base_primitive = primitive["fun_info"]
    in_attribute = primitive["att_input"]
    map_rule = primitive["map_rule"]
    primitive_dictionary = dict()

    # Dynamically import modules and functions needed to eval input/output types
    custom_globals = {}
    for module_name, function_names in imports.items():
        module = importlib.import_module(module_name)
        for function_name in function_names:
            custom_globals[function_name] = getattr(module, function_name)

    def eval_with_globals(expression):
        return eval(expression, custom_globals)

    for in_category in in_attribute["category"]:
        for in_dim in in_attribute["dimension"]:
            for in_rank in in_attribute["rank"]:
                # compute the primitive name taking into account
                # the right category, dim and rank
                in_rank = in_rank.replace("SC", "")
                primitive_name = base_primitive["name"] + in_category + in_dim + in_rank
                in_type_name = []
                # compute the input type list
                for i, input in enumerate(primitive["input"]):
                    # float type must be handled separately
                    if input == "float":
                        in_type_name.append(input)
                    elif len(in_rank) == 2:
                        # in this case the correct rank must be taken
                        in_type_name.append(input + in_category + in_dim + in_rank[i])
                    else:
                        in_type_name.append(input + in_category + in_dim + in_rank)
                in_type = list(map(eval_with_globals, in_type_name))
                out_category = map_rule["category"](in_category)
                out_dim = str(map_rule["dimension"](int(in_dim)))
                out_rank = map_rule["rank"](in_rank)
                out_type_name = primitive["output"] + out_category + out_dim + out_rank
                out_type = eval_with_globals(out_type_name)
                primitive_dictionary[primitive_name] = PrimitiveParams(
                    base_primitive["fun"], in_type, out_type
                )
    return primitive_dictionary


def add_primitives_to_pset(
    pset: PrimitiveSetTyped, primitives_to_add: list, primitives_collection: dict
):
    """Add a given list of primitives to a given PrimitiveSet.

    Args:
        pset: a primitive set.
        pset_primitives: list of primitives to be added. Each primitive is encoded
            as a dictionary composed of three keys: 'name', containing the name of
            the general primitive (e.g. cob for the coboundary); dimension', containing
            a list of the possible dimensions of the primitive input (or None if a
            scalar primitive is considered); 'rank', containing a list of the possible
            ranks of the primitive input (or None if a scalar primitive is considered).
    """
    for primitive in primitives_to_add:
        # pre-process scalar primitives
        if primitive["dimension"] is None:
            primitive["dimension"] = []
        if primitive["rank"] is None:
            primitive["rank"] = []
        # save dimensions and ranks not admitted for the problem
        non_feasible_dimensions = list(
            set(("0", "1", "2")) - set(primitive["dimension"])
        )
        non_feasible_ranks = list(set(("SC", "V", "T")) - set(primitive["rank"]))
        # iterate over all the primitives, pre-computed and stored in the dictionary
        # primitives
        for typed_primitive in primitives_collection.keys():
            if primitive["name"] in typed_primitive:
                # remove the case in which the name of the primitive is a subname
                # of type_primitive (e.g. if primitive['name'] = sin and typed_primitive
                # = arcsin, we don't want to add the primitive)
                exact_name_check = (
                    len(typed_primitive.replace(primitive["name"], "")) <= 2
                )
                # check if the dimension/rank of a typed primitive
                # is admissible, i.e. if it does not coincide with a non-admissible
                # dimension/rank
                # FIXME: change this!
                check_wrong_dim_primal = sum(
                    [
                        typed_primitive.count("P" + obj)
                        for obj in non_feasible_dimensions
                    ]
                )
                check_wrong_dim_dual = sum(
                    [
                        typed_primitive.count("D" + obj)
                        for obj in non_feasible_dimensions
                    ]
                )
                check_rank = sum(
                    [typed_primitive.count("P" + obj) for obj in non_feasible_ranks]
                )
                check_wrong_dim_rank = (
                    check_wrong_dim_primal + check_wrong_dim_dual + check_rank
                )
                if check_wrong_dim_rank == 0 and exact_name_check:
                    op = primitives_collection[typed_primitive].op
                    in_types = primitives_collection[typed_primitive].in_types
                    out_type = primitives_collection[typed_primitive].out_type
                    pset.addPrimitive(op, in_types, out_type, name=typed_primitive)


def deap_primitive_to_sympy_expr(prim: Primitive, conversion_rules: Dict, args: Tuple):
    """Convert a deap scalar primitive into a sympy primitive.

    Args:
        prim: the primitive.
        conversion_rules: a dictionary of conversion rules.
        args: args of the primitive.

    Returns:
        the converted primitive.

    """
    prim_formatter = conversion_rules.get(prim.name, prim.format)

    return prim_formatter(*args)


def stringify_for_sympy(
    f: PrimitiveTree, conversion_rules: Dict, special_term_name: str
) -> str:
    """Returns a sympy-compatible expression.

    Args:
        f: the individual tree (DEAP format)
        conversion_rules: a dictionary of conversion rules.
        special_term_name: name of the constant placeholder.

    Returns:
        the sympy-compatible expression.
    """
    prim_string = ""
    stack = []
    const_idx = 0
    for node in f:
        stack.append((node, []))
        while len(stack[-1][1]) == stack[-1][0].arity:
            prim, args = stack.pop()
            if prim.name == special_term_name:
                # substitute the c placeholder with the constant value
                prim_string = f.consts[const_idx]
                # update the constant index
                const_idx += 1
            else:
                prim_string = deap_primitive_to_sympy_expr(prim, conversion_rules, args)
            if len(stack) == 0:
                break  # If stack is empty, all nodes should have been seen
            stack[-1][1].append(prim_string)
    return prim_string
