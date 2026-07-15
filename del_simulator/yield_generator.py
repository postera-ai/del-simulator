from abc import ABC, abstractmethod
from rdkit import Chem
import numpy as np
import typing as T
from functools import lru_cache, partial
from rdkit.Chem import rdMolHash
import logging
import random
import hashlib

YIELD_PRECISION = 6


class YieldGenerator(ABC):
    def __init__(
        self, jitter_sigma: float = 0.0, jitter_seed: int = 42, *args, **kwargs
    ) -> None:
        self.jitter_sigma = jitter_sigma
        self.jitter_rng = np.random.default_rng(seed=jitter_seed)

    def add_jitter(self, yield_: float) -> float:
        jittered = yield_ + self.jitter_rng.normal(loc=0.0, scale=self.jitter_sigma)
        return float(np.clip(jittered, 0.0, 1.0))

    @abstractmethod
    def generate_yield(
        self,
        product: Chem.Mol,
        existing_molecule: Chem.Mol,
        added_building_block: Chem.Mol | None,
    ) -> float:
        """
        Return the predicted yield for a given reaction, reactants, and product.

        This is restricted to reactions with at most two reactants

        Within the DEL simulator, we limit ourselves to reactions with at most two components -- the DNA-attached portion
        and the new additional building block (or nothing, in case of deprotection reactions)

        """

        raise NotImplementedError


class YieldGeneratorUniform(YieldGenerator):
    """
    A basic yield generator that return a uniform yield for all products.
    """

    def __init__(
        self, success_fraction: float, jitter_sigma: float = 0.0, jitter_seed: int = 42
    ):
        assert 0.0 <= success_fraction <= 1.0

        super().__init__(jitter_sigma, jitter_seed)
        self.success_fraction = success_fraction

    @lru_cache(maxsize=1000)
    def generate_yield(
        self,
        product: Chem.Mol,
        current_mol: Chem.Mol,
        bb_to_add: Chem.Mol | None,
    ) -> float:
        return self.add_jitter(self.success_fraction)


class YieldGeneratorBeta(YieldGenerator):
    def __init__(
        self,
        alpha: float = 10.0,
        beta: float = 1.0,
        jitter_sigma: float = 0.0,
        jitter_seed: int = 42,
        hash_product: bool = False,
        hash_current_mol: bool = False,
        hash_bb_to_add: bool = True,
    ):
        assert alpha > 0 and beta > 0

        super().__init__(jitter_sigma, jitter_seed)

        self.alpha = alpha
        self.beta = beta

        self.hash_product = hash_product
        self.hash_current_mol = hash_current_mol
        self.hash_bb_to_add = hash_bb_to_add

    @lru_cache(maxsize=1000)
    def hash_mols_to_seed(
        self,
        current_mol: Chem.Mol,
        bb_to_add: Chem.Mol,
        product: Chem.Mol,
    ) -> int:
        hash_function = rdMolHash.HashFunction.MolFormula
        combined_hash = ""

        if self.hash_bb_to_add:
            if bb_to_add is None:
                raise ValueError("No building block to hash!")
            combined_hash += rdMolHash.MolHash(bb_to_add, hash_function)
        if self.hash_current_mol:
            combined_hash += rdMolHash.MolHash(current_mol, hash_function)
        if self.hash_product:
            combined_hash += rdMolHash.MolHash(product, hash_function)

        # fixme is this slow?
        seed = int(
            hashlib.md5((combined_hash).encode(encoding="UTF-8")).hexdigest(),
            16,
        )

        return seed

    @lru_cache(maxsize=1000)
    def generate_yield(
        self,
        product: Chem.Mol,
        current_mol: Chem.Mol,
        bb_to_add: Chem.Mol | None,
    ) -> float:
        seed = self.hash_mols_to_seed(
            current_mol,
            bb_to_add,
            product,
        )

        rng = np.random.default_rng(seed=seed)
        probability_rxn = self.add_jitter(rng.beta(self.alpha, self.beta))

        return probability_rxn


class ReactionYieldsGenerator:
    def __init__(
        self,
        yield_generator: YieldGenerator | None = None,
    ):
        """A class to return the predicted yields for all products of a reaction, including the unreacted starting material.

        It applies a yield_generator to all possible products of a reaction.

        """
        self.yield_generator = yield_generator or YieldGeneratorUniform(1.0)

    def normalize_yields(
        self, yields: np.ndarray[(1,), np.float64]
    ) -> np.ndarray[(1,), np.float64]:
        """
        Normalize the yield vector so the no-product yield[0] and sum(yield[1:]) add up to 1.

        If there are no products, return [1.0].
        If there is >1 product, the total reacted probability is mean(yields[1:]),
        split across products in proportion to their relative raw yields. This keeps
        the total reacted fraction pinned at mean(yields[1:]) regardless of how many
        distinct products RDKit happens to enumerate for a given reaction.

        """
        if len(yields) == 1:
            logging.debug(
                "Only one product present in the yield! Since the no-product is always added, the reaction has failed!"
            )
            return np.array([1.0])

        # Defensive: yield generators are expected to return values in [0, 1], but a
        # generator that adds unclamped noise could violate that, which would otherwise
        # produce a negative "no-reaction" probability or negative product shares below.
        raw_product_yields = np.clip(yields[1:], 0.0, 1.0)
        mean_yield = np.mean(
            raw_product_yields
        )  # probability some product forms, on average
        total_raw = np.sum(raw_product_yields)

        if total_raw > 0:
            product_shares = raw_product_yields / total_raw * mean_yield
        else:
            product_shares = raw_product_yields  # all zero

        return np.concatenate(([1.0 - mean_yield], product_shares))

    def generate_yields(
        self,
        current_mol: Chem.Mol,
        bb_to_add: Chem.Mol | None,
        products: T.List[Chem.Mol],
    ) -> np.ndarray[(1,), np.float64]:
        """
        Generate yields for a given set of reactants and products, using the provided yieldgenerator

        Args:
            reactant1 (Chem.Mol ): The first reactant molecule.
            reactant2 (Chem.Mol | None): The second reactant molecule.
            products (List[Chem.Mol]): The list of product molecules.
            rxn (ChemicalReaction): The chemical reaction.

        Returns:
            np.ndarray[(1,), np.float64]: An array of yields for each product.

        """
        rxn_yields = []

        _generate_yield = partial(
            self.yield_generator.generate_yield,
            current_mol=current_mol,
            bb_to_add=bb_to_add,
        )

        # By convention, the first element of the products array is the unreacted starting product
        rxn_yields = np.array(
            [0.0] + list(map(_generate_yield, products[1:])), dtype=np.float64
        )

        rxn_yields = np.round(self.normalize_yields(rxn_yields), YIELD_PRECISION)

        return rxn_yields
