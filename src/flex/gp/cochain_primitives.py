from dctkit.dec import cochain as C
import operator
from functools import partial
import jax.numpy
from .primitives import switch_category, generate_primitive_variants

# Define the modules and functions needed to eval inputs and outputs
modules_functions = {"dctkit.dec": ["cochain"]}


def inv_scalar_mul(c, f):
    try:
        return C.scalar_mul(c, 1 / f)
    except ZeroDivisionError:
        return C.scalar_mul(c, jax.numpy.nan)


# define cochain primitives
add_coch = {
    "fun_info": {"name": "AddC", "fun": C.add},
    "input": ["cochain.Cochain", "cochain.Cochain"],
    "output": "cochain.Cochain",
    "att_input": {
        "category": ("P", "D"),
        "dimension": ("0", "1", "2"),
        "rank": ("SC", "V", "T"),
    },
    "map_rule": {
        "category": lambda x: x,
        "dimension": lambda x: x,
        "rank": lambda x: x,
    },
}
sub_coch = {
    "fun_info": {"name": "SubC", "fun": C.sub},
    "input": ["cochain.Cochain", "cochain.Cochain"],
    "output": "cochain.Cochain",
    "att_input": {
        "category": ("P", "D"),
        "dimension": ("0", "1", "2"),
        "rank": ("SC", "V", "T"),
    },
    "map_rule": {
        "category": lambda x: x,
        "dimension": lambda x: x,
        "rank": lambda x: x,
    },
}
coboundary = {
    "fun_info": {"name": "cob", "fun": C.coboundary},
    "input": ["cochain.Cochain"],
    "output": "cochain.Cochain",
    "att_input": {
        "category": ("P", "D"),
        "dimension": ("0", "1"),
        "rank": ("SC", "V", "T"),
    },
    "map_rule": {
        "category": lambda x: x,
        "dimension": partial(operator.add, 1),
        "rank": lambda x: x,
    },
}
codifferential = {
    "fun_info": {"name": "del", "fun": C.codifferential},
    "input": ["cochain.Cochain"],
    "output": "cochain.Cochain",
    "att_input": {
        "category": ("P", "D"),
        "dimension": ("1", "2"),
        "rank": ("SC", "V", "T"),
    },
    "map_rule": {
        "category": lambda x: x,
        "dimension": partial(operator.add, -1),
        "rank": lambda x: x,
    },
}
tr_coch = {
    "fun_info": {"name": "tr", "fun": C.trace},
    "input": ["cochain.Cochain"],
    "output": "cochain.Cochain",
    "att_input": {"category": ("P", "D"), "dimension": ("0", "1", "2"), "rank": ("T",)},
    "map_rule": {
        "category": lambda x: x,
        "dimension": lambda x: x,
        "rank": lambda x: "",
    },
}
mul_FT = {
    "fun_info": {"name": "MF", "fun": C.scalar_mul},
    "input": ["cochain.Cochain", "float"],
    "output": "cochain.Cochain",
    "att_input": {
        "category": ("P", "D"),
        "dimension": ("0", "1", "2"),
        "rank": ("SC", "V", "T"),
    },
    "map_rule": {
        "category": lambda x: x,
        "dimension": lambda x: x,
        "rank": lambda x: x,
    },
}
inv_mul_FT = {
    "fun_info": {"name": "InvM", "fun": inv_scalar_mul},
    "input": ["cochain.Cochain", "float"],
    "output": "cochain.Cochain",
    "att_input": {
        "category": ("P", "D"),
        "dimension": ("0", "1", "2"),
        "rank": ("SC", "V", "T"),
    },
    "map_rule": {
        "category": lambda x: x,
        "dimension": lambda x: x,
        "rank": lambda x: x,
    },
}
mul_coch = {
    "fun_info": {"name": "CMul", "fun": C.cochain_mul},
    "input": ["cochain.Cochain", "cochain.Cochain"],
    "output": "cochain.Cochain",
    "att_input": {
        "category": ("P", "D"),
        "dimension": ("0", "1", "2"),
        "rank": ("SC",),
    },
    "map_rule": {
        "category": lambda x: x,
        "dimension": lambda x: x,
        "rank": lambda x: x,
    },
}
tran_coch = {
    "fun_info": {"name": "tran", "fun": C.transpose},
    "input": ["cochain.Cochain"],
    "output": "cochain.Cochain",
    "att_input": {"category": ("P", "D"), "dimension": ("0", "1", "2"), "rank": ("T",)},
    "map_rule": {
        "category": lambda x: x,
        "dimension": lambda x: x,
        "rank": lambda x: x,
    },
}
sym_coch = {
    "fun_info": {"name": "sym", "fun": C.sym},
    "input": ["cochain.Cochain"],
    "output": "cochain.Cochain",
    "att_input": {"category": ("P", "D"), "dimension": ("0", "1", "2"), "rank": ("T",)},
    "map_rule": {
        "category": lambda x: x,
        "dimension": lambda x: x,
        "rank": lambda x: x,
    },
}
star_1 = {
    "fun_info": {"name": "St1", "fun": C.star},
    "input": ["cochain.Cochain"],
    "output": "cochain.Cochain",
    "att_input": {
        "category": ("P", "D"),
        "dimension": ("0", "1"),
        "rank": ("SC", "V", "T"),
    },
    "map_rule": {
        "category": partial(switch_category, ("P", "D")),
        "dimension": partial(lambda x, y: y - x, y=1),
        "rank": lambda x: x,
    },
}
star_2 = {
    "fun_info": {"name": "St2", "fun": C.star},
    "input": ["cochain.Cochain"],
    "output": "cochain.Cochain",
    "att_input": {
        "category": ("P", "D"),
        "dimension": ("0", "1", "2"),
        "rank": ("SC", "V", "T"),
    },
    "map_rule": {
        "category": partial(switch_category, ("P", "D")),
        "dimension": partial(lambda x, y: y - x, y=2),
        "rank": lambda x: x,
    },
}
inner_product = {
    "fun_info": {"name": "Inn", "fun": C.inner},
    "input": ["cochain.Cochain", "cochain.Cochain"],
    "output": "float",
    "att_input": {
        "category": ("P", "D"),
        "dimension": ("0", "1", "2"),
        "rank": ("SC", "V", "T"),
    },
    "map_rule": {
        "category": lambda x: "",
        "dimension": lambda x: "",
        "rank": lambda x: "",
    },
}
sin_coch = {
    "fun_info": {"name": "Sin", "fun": C.sin},
    "input": ["cochain.Cochain"],
    "output": "cochain.Cochain",
    "att_input": {
        "category": ("P", "D"),
        "dimension": ("0", "1", "2"),
        "rank": ("SC", "V", "T"),
    },
    "map_rule": {
        "category": lambda x: x,
        "dimension": lambda x: x,
        "rank": lambda x: x,
    },
}
arcsin_coch = {
    "fun_info": {"name": "ArcSin", "fun": C.arcsin},
    "input": ["cochain.Cochain"],
    "output": "cochain.Cochain",
    "att_input": {
        "category": ("P", "D"),
        "dimension": ("0", "1", "2"),
        "rank": ("SC", "V", "T"),
    },
    "map_rule": {
        "category": lambda x: x,
        "dimension": lambda x: x,
        "rank": lambda x: x,
    },
}
cos_coch = {
    "fun_info": {"name": "Cos", "fun": C.cos},
    "input": ["cochain.Cochain"],
    "output": "cochain.Cochain",
    "att_input": {
        "category": ("P", "D"),
        "dimension": ("0", "1", "2"),
        "rank": ("SC", "V", "T"),
    },
    "map_rule": {
        "category": lambda x: x,
        "dimension": lambda x: x,
        "rank": lambda x: x,
    },
}
arccos_coch = {
    "fun_info": {"name": "ArcCos", "fun": C.arccos},
    "input": ["cochain.Cochain"],
    "output": "cochain.Cochain",
    "att_input": {
        "category": ("P", "D"),
        "dimension": ("0", "1", "2"),
        "rank": ("SC", "V", "T"),
    },
    "map_rule": {
        "category": lambda x: x,
        "dimension": lambda x: x,
        "rank": lambda x: x,
    },
}
exp_coch = {
    "fun_info": {"name": "Exp", "fun": C.exp},
    "input": ["cochain.Cochain"],
    "output": "cochain.Cochain",
    "att_input": {
        "category": ("P", "D"),
        "dimension": ("0", "1", "2"),
        "rank": ("SC", "V", "T"),
    },
    "map_rule": {
        "category": lambda x: x,
        "dimension": lambda x: x,
        "rank": lambda x: x,
    },
}
log_coch = {
    "fun_info": {"name": "Log", "fun": C.log},
    "input": ["cochain.Cochain"],
    "output": "cochain.Cochain",
    "att_input": {
        "category": ("P", "D"),
        "dimension": ("0", "1", "2"),
        "rank": ("SC", "V", "T"),
    },
    "map_rule": {
        "category": lambda x: x,
        "dimension": lambda x: x,
        "rank": lambda x: x,
    },
}
sqrt_coch = {
    "fun_info": {"name": "Sqrt", "fun": C.sqrt},
    "input": ["cochain.Cochain"],
    "output": "cochain.Cochain",
    "att_input": {
        "category": ("P", "D"),
        "dimension": ("0", "1", "2"),
        "rank": ("SC", "V", "T"),
    },
    "map_rule": {
        "category": lambda x: x,
        "dimension": lambda x: x,
        "rank": lambda x: x,
    },
}
square_coch = {
    "fun_info": {"name": "Square", "fun": C.square},
    "input": ["cochain.Cochain"],
    "output": "cochain.Cochain",
    "att_input": {
        "category": ("P", "D"),
        "dimension": ("0", "1", "2"),
        "rank": ("SC", "V", "T"),
    },
    "map_rule": {
        "category": lambda x: x,
        "dimension": lambda x: x,
        "rank": lambda x: x,
    },
}

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
