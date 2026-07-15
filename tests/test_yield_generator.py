import numpy as np
import pytest
from rdkit import Chem

from del_simulator.yield_generator import (
    YieldGeneratorBeta,
    YieldGeneratorUniform,
    ReactionYieldsGenerator,
)


def _mol(smiles):
    return Chem.MolFromSmiles(smiles)


def test_generate_yield_is_deterministic_given_same_inputs():
    generator = YieldGeneratorBeta(alpha=10.0, beta=1.0, jitter_sigma=0.0)
    product, current_mol, bb_to_add = _mol("CCO"), _mol("CC"), _mol("O")

    first = generator.generate_yield(product, current_mol, bb_to_add)
    second = generator.generate_yield(product, current_mol, bb_to_add)

    assert first == second
    assert 0.0 <= first <= 1.0


def test_generate_yield_differs_with_different_bb_to_add():
    """
    hash_bb_to_add=True by default, so the seed fed to the beta distribution is derived
    from bb_to_add's formula -- swapping it for an unrelated building block should (with
    overwhelming probability) change the sampled yield.
    """
    generator = YieldGeneratorBeta(alpha=10.0, beta=1.0, jitter_sigma=0.0)
    product, current_mol = _mol("CCO"), _mol("CC")

    yield_with_o = generator.generate_yield(product, current_mol, _mol("O"))
    yield_with_n = generator.generate_yield(product, current_mol, _mol("N"))

    assert yield_with_o != yield_with_n


def test_hash_mols_to_seed_raises_without_bb_to_add():
    generator = YieldGeneratorBeta(hash_bb_to_add=True)

    with pytest.raises(ValueError, match="No building block to hash"):
        generator.generate_yield(_mol("CCO"), _mol("CC"), None)


def test_add_jitter_clamps_to_valid_probability_range():
    """
    add_jitter used to add unclamped Gaussian noise, so a yield near 0 or 1 plus a large
    enough jitter_sigma could push the result outside [0, 1] -- an invalid probability that
    then corrupts normalize_yields' downstream math (see test_normalize_yields_* below).
    """
    generator = YieldGeneratorUniform(success_fraction=0.99, jitter_sigma=10.0)

    for _ in range(50):
        jittered = generator.add_jitter(generator.success_fraction)
        assert 0.0 <= jittered <= 1.0


def test_normalize_yields_clips_out_of_range_inputs():
    """
    Simulates what an unclamped jittered yield generator used to be able to produce: a raw
    product yield above 1.0. Before clipping, this drove mean_yield above 1.0, making the
    "no reaction" share (1.0 - mean_yield) negative -- an invalid probability written
    straight into library.csv's relative_fraction column.
    """
    rxn_yields_gen = ReactionYieldsGenerator(YieldGeneratorUniform(1.0))

    result = rxn_yields_gen.normalize_yields(np.array([0.0, 1.5, 1.3]))

    assert (result >= 0.0).all()
    assert np.isclose(result.sum(), 1.0, atol=1e-5)
