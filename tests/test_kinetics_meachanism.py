import numpy as np

from vr.kinetics import Mechanism


def test_van_de_vusse_mechanism():

    species = ["A", "B", "C", "D"]

    stoichiometry = [
        [-1,  0, -2],
        [ 1, -1,  0],
        [ 0,  1,  0],
        [ 0,  0,  1],
    ]

    reaction_orders = [
        [1, 0, 2],
        [0, 1, 0],
        [0, 0, 0],
        [0, 0, 0],
    ]

    A = [1.0e3, 5.0e2, 2.0e2]
    Ea = [40000, 50000, 45000]

    mechanism = Mechanism(
        species=species,
        stoichiometry=stoichiometry,
        reaction_orders=reaction_orders,
        pre_exponential_factors=A,
        activation_energies=Ea,
    )

    concentrations = [1.0, 0.2, 0.0, 0.0]
    temperature = 350.0

    source = mechanism.reaction_source(
        concentrations,
        temperature,
    )

    assert isinstance(source, np.ndarray)
    assert source.shape == (4,)
    assert np.all(np.isfinite(source))