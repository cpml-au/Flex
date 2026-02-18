from deap import gp
from dctkit.dec import cochain as C
from flex.gp.primitives import add_primitives_to_pset_from_dict


def test_primitives(set_test_dir):
    # define primitives dictionaries
    scalar_primitives_jax = {
        "imports": {"flex.gp.jax_primitives": ["jax_primitives"]},
        "used": [
            {"name": "AddF", "dimension": "None", "rank": "None"},
            {"name": "SubF", "dimension": "None", "rank": "None"},
            {"name": "MulF", "dimension": "None", "rank": "None"},
            {"name": "Div", "dimension": "None", "rank": "None"},
            {"name": "SquareF", "dimension": "None", "rank": "None"},
            {"name": "SinF", "dimension": "None", "rank": "None"},
            {"name": "ArcsinF", "dimension": "None", "rank": "None"},
            {"name": "CosF", "dimension": "None", "rank": "None"},
            {"name": "ArccosF", "dimension": "None", "rank": "None"},
            {"name": "ExpF", "dimension": "None", "rank": "None"},
            {"name": "LogF", "dimension": "None", "rank": "None"},
            {"name": "InvF", "dimension": "None", "rank": "None"},
        ],
    }
    scalar_primitives_numpy = {
        "imports": {"flex.gp.numpy_primitives": ["numpy_primitives"]},
        "used": [
            {"name": "add", "dimension": "None", "rank": "None"},
            {"name": "sub", "dimension": "None", "rank": "None"},
            {"name": "mul", "dimension": "None", "rank": "None"},
            {"name": "div", "dimension": "None", "rank": "None"},
            {"name": "aq", "dimension": "None", "rank": "None"},
            {"name": "square", "dimension": "None", "rank": "None"},
            {"name": "sin", "dimension": "None", "rank": "None"},
            {"name": "cos", "dimension": "None", "rank": "None"},
            {"name": "exp", "dimension": "None", "rank": "None"},
            {"name": "log", "dimension": "None", "rank": "None"},
        ],
    }

    cochain_primitives_scalar = {
        "imports": {"flex.gp.cochain_primitives": ["coch_primitives"]},
        "used": [
            {"name": "AddC", "dimension": ["0", "1"], "rank": ["SC"]},
            {"name": "SubC", "dimension": ["0", "1"], "rank": ["SC"]},
            {"name": "cob", "dimension": ["0", "1"], "rank": ["SC"]},
            {"name": "del", "dimension": ["1"], "rank": ["SC"]},
            {"name": "St1", "dimension": ["0", "1"], "rank": ["SC"]},
            {"name": "MF", "dimension": ["0", "1"], "rank": ["SC"]},
            {"name": "Inn", "dimension": ["0", "1"], "rank": ["SC"]},
        ],
    }

    cochain_primitives_vector_tensor = {
        "imports": {"flex.gp.cochain_primitives": ["coch_primitives"]},
        "used": [
            {"name": "AddC", "dimension": ["0", "1", "2"], "rank": ["SC", "V", "T"]},
            {"name": "SubC", "dimension": ["0", "1", "2"], "rank": ["SC", "V", "T"]},
            {"name": "cob", "dimension": ["0", "1", "2"], "rank": ["SC"]},
            {"name": "del", "dimension": ["2", "1"], "rank": ["SC"]},
            {"name": "St2", "dimension": ["0", "1", "2"], "rank": ["SC", "V", "T"]},
            {"name": "MF", "dimension": ["0", "1", "2"], "rank": ["SC", "V", "T"]},
            {"name": "Inn", "dimension": ["0", "1", "2"], "rank": ["SC", "V", "T"]},
            {"name": "tr", "dimension": ["0", "1", "2"], "rank": ["T"]},
            {"name": "tran", "dimension": ["0", "1", "2"], "rank": ["T"]},
            {"name": "sym", "dimension": ["0", "1", "2"], "rank": ["T"]},
        ],
    }

    # define dummy primitive sets just to test the primitive functions
    scalar_pset_jax = gp.PrimitiveSetTyped("main", [float], float)
    scalar_pset_numpy = gp.PrimitiveSetTyped("main", [float], float)
    pset_cochain_scalar = gp.PrimitiveSetTyped("main", [C.Cochain], C.Cochain)
    pset_cochain_vector_tensor = gp.PrimitiveSetTyped("main", [C.Cochain], C.Cochain)

    scalar_pset_jax = add_primitives_to_pset_from_dict(
        scalar_pset_jax, scalar_primitives_jax
    )
    scalar_pset_numpy = add_primitives_to_pset_from_dict(
        scalar_pset_numpy, scalar_primitives_numpy
    )
    pset_cochain_scalar = add_primitives_to_pset_from_dict(
        pset_cochain_scalar, cochain_primitives_scalar
    )
    pset_cochain_vector_tensor = add_primitives_to_pset_from_dict(
        pset_cochain_vector_tensor, cochain_primitives_vector_tensor
    )

    all_primitives_jax = scalar_pset_jax.primitives[scalar_pset_jax.ret]
    all_primitives_numpy = scalar_pset_numpy.primitives[scalar_pset_numpy.ret]
    # for cochains there are multiple return types
    return_types_cochain_scalar = [
        C.CochainP0,
        C.CochainP1,
        C.CochainP2,
        C.CochainD0,
        C.CochainD1,
        C.CochainD2,
    ]
    return_types_cochain_vector_tensor = [
        C.CochainP0,
        C.CochainP1,
        C.CochainP2,
        C.CochainD0,
        C.CochainD1,
        C.CochainD2,
        C.CochainP0V,
        C.CochainP1V,
        C.CochainP2V,
        C.CochainD0V,
        C.CochainD1V,
        C.CochainD2V,
        C.CochainP0T,
        C.CochainP1T,
        C.CochainP2T,
        C.CochainD0T,
        C.CochainD1T,
        C.CochainD2T,
    ]

    all_primitives_cochain_scalar = []
    for ret in return_types_cochain_scalar:
        all_primitives_cochain_scalar.extend(pset_cochain_scalar.primitives[ret])
    all_primitives_cochain_vector_tensor = []
    for ret in return_types_cochain_vector_tensor:
        all_primitives_cochain_vector_tensor.extend(
            pset_cochain_vector_tensor.primitives[ret]
        )

    primitive_names_jax = [p.name for p in all_primitives_jax]
    primitive_names_numpy = [p.name for p in all_primitives_numpy]
    primitive_names_cochain_scalar = [p.name for p in all_primitives_cochain_scalar]
    primitive_names_cochain_vector_tensor = [
        p.name for p in all_primitives_cochain_vector_tensor
    ]

    true_primitive_names_jax = [
        "AddF",
        "SubF",
        "MulF",
        "Div",
        "SquareF",
        "SinF",
        "ArcsinF",
        "CosF",
        "ArccosF",
        "ExpF",
        "LogF",
        "InvF",
    ]

    true_primitive_names_numpy = [
        "add",
        "sub",
        "mul",
        "div",
        "aq",
        "square",
        "sin",
        "cos",
        "exp",
        "log",
    ]
    true_primitive_names_cochain_scalar = [
        "AddCP0",
        "SubCP0",
        "delP1",
        "MFP0",
        "AddCP1",
        "SubCP1",
        "cobP0",
        "MFP1",
        "cobP1",
        "AddCD0",
        "SubCD0",
        "delD1",
        "MFD0",
        "AddCD1",
        "SubCD1",
        "cobD0",
        "MFD1",
        "cobD1",
    ]
    true_primitive_names_cochain_vector_tensor = [
        "AddCP0",
        "SubCP0",
        "delP1",
        "MFP0",
        "trP0T",
        "AddCP1",
        "SubCP1",
        "cobP0",
        "delP2",
        "MFP1",
        "trP1T",
        "AddCP2",
        "SubCP2",
        "cobP1",
        "MFP2",
        "trP2T",
        "AddCD0",
        "SubCD0",
        "delD1",
        "MFD0",
        "trD0T",
        "AddCD1",
        "SubCD1",
        "cobD0",
        "delD2",
        "MFD1",
        "trD1T",
        "AddCD2",
        "SubCD2",
        "cobD1",
        "MFD2",
        "trD2T",
        "AddCP0V",
        "SubCP0V",
        "MFP0V",
        "AddCP1V",
        "SubCP1V",
        "MFP1V",
        "AddCP2V",
        "SubCP2V",
        "MFP2V",
        "AddCD0V",
        "SubCD0V",
        "MFD0V",
        "AddCD1V",
        "SubCD1V",
        "MFD1V",
        "AddCD2V",
        "SubCD2V",
        "MFD2V",
        "AddCP0T",
        "SubCP0T",
        "MFP0T",
        "tranP0T",
        "symP0T",
        "AddCP1T",
        "SubCP1T",
        "MFP1T",
        "tranP1T",
        "symP1T",
        "AddCP2T",
        "SubCP2T",
        "MFP2T",
        "tranP2T",
        "symP2T",
        "AddCD0T",
        "SubCD0T",
        "MFD0T",
        "tranD0T",
        "symD0T",
        "AddCD1T",
        "SubCD1T",
        "MFD1T",
        "tranD1T",
        "symD1T",
        "AddCD2T",
        "SubCD2T",
        "MFD2T",
        "tranD2T",
        "symD2T",
    ]

    # NOTE: the order of the list does not matter
    assert set(primitive_names_jax) == set(true_primitive_names_jax)
    assert set(primitive_names_numpy) == set(true_primitive_names_numpy)
    assert set(primitive_names_cochain_scalar) == set(
        true_primitive_names_cochain_scalar
    )
    assert set(primitive_names_cochain_vector_tensor) == set(
        true_primitive_names_cochain_vector_tensor
    )
