from rdkit import Chem
from rdkit import DataStructs
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
import typing as T
from pandarallel import pandarallel
import logging
from tqdm import tqdm
from scipy import stats
from del_simulator.utils.utils import (
    FingerprintFeaturizer,
)

from del_simulator.core import (
    SMILES,
    Library,
    KernelParameters,
    TverskyParameters,
    FingerprintGeneratorMethod,
    MorganFingerPrintGeneratorParameters,
)

from functools import lru_cache
from joblib import Parallel, delayed
import hashlib


class AffinityCalculator(ABC):
    @abstractmethod
    def get_affinities(self, query_library: Library) -> (np.array, np.array):
        return


# FIXME why is this a mixin anyways?
class FingerprintAndSimilarityMixin:
    def __init__(self, parallelism: int = 4, progress_bar: bool = False) -> None:
        pandarallel.initialize(progress_bar=progress_bar, nb_workers=parallelism)

    def compute_fingerprints(self, smiles: T.List[SMILES]):  # -> T.List[np.ndarray]:
        """
        Compute fingerprints on a list of smiles
        """
        # _df = pd.DataFrame(smiles, columns=["smiles"])
        fps = [self.fpf.process_smiles(x) for x in smiles]
        return fps

    @staticmethod
    def compute_tversky_similarity_mx(
        reference_fingerprints,
        query_fingerprints,
        a: float = 1.0,
        b: float = 0.0,
    ):
        # compute the similarity matrix between the reference and the query smiles

        sim_mx = np.zeros((len(query_fingerprints), len(reference_fingerprints)))

        for idx, ref_fp in enumerate(reference_fingerprints):
            # if the reference molecules are known hits that are smaller than the DEL members;
            # then the asymmetric similarity measure is more appropriate (i.e. a=1, b=0)
            # see https://en.wikipedia.org/wiki/Tversky_index and https://www.daylight.com/dayhtml/doc/theory/theory.finger.html

            sim_mx[:, idx] = DataStructs.BulkTverskySimilarity(
                ref_fp, query_fingerprints, a=a, b=b
            )

        return sim_mx


class TruncatedNormalSampler:
    def __init__(
        self,
        loc: float = 5.0,
        scale: float = 2.0,
        truncated_min: float = 0,
        truncated_max: float = 8,
        seed: int = 42,
    ):
        self.loc = loc
        self.scale = scale
        self.truncated_min = truncated_min
        self.truncated_max = truncated_max
        self.seed = seed

    @lru_cache(maxsize=1000)
    def sample(self, seed: int = 42):
        a = (
            self.truncated_min - self.loc
        ) / self.scale  # FIXME implement upper-bound trunctation
        aff = stats.truncnorm.rvs(
            a=a,
            b=np.inf,
            loc=self.loc,
            scale=self.scale,
            random_state=(seed + self.seed) % (2**32),
        )
        return aff


class LogNormalSampler:
    def __init__(
        self,
        s: float = 1.0,
        loc: float = 0.0,
        scale: float = 1.0,
        seed: int = 42,
    ):
        self.s = s
        self.loc = loc
        self.scale = scale
        self.seed = seed

    @lru_cache(maxsize=1000)
    def sample(self, seed: int = 42):
        aff = stats.lognorm.rvs(
            s=self.s,
            loc=self.loc,
            scale=self.scale,
            random_state=(seed + self.seed) % (2**32),
        )
        return aff


class RandomAffinityCalculator(AffinityCalculator):
    def __init__(
        self,
        sampler,
        parallelism: int = 4,
    ) -> None:
        super().__init__()
        self.parallelism = parallelism
        self.sampler = sampler

    def _get_affinities(self, smiles: T.List[SMILES]) -> np.array:
        affinities = []
        for s in smiles:
            seed = int(
                hashlib.md5((s).encode(encoding="UTF-8")).hexdigest(),
                16,
            ) % (2**32)

            affinities.append(
                self.sampler.sample(seed=seed)
            )  # fixme expose the distribution interface
        affinities = np.array(affinities)

        logging.info("Computed %s unique affinity values", len(affinities))
        return affinities

    def get_affinities(self, query_library: Library, chunksize: int = 8192) -> np.array:
        library_size = len(query_library.nsynthon_id)
        # sorted (not just set()) so unique_smiles has a stable, run-to-run reproducible
        # order -- callers such as SimilarityAffinityCalculator index unique_affinities
        # by position and rely on that order matching their own sorted(set(...)) call.
        unique_smiles = sorted(set(query_library.smiles))
        num_unique_smiles = len(unique_smiles)
        num_chunks = (num_unique_smiles - 1) // chunksize + 1

        logging.info(
            f"Computing Distribution Affinities for {library_size} smiles({num_unique_smiles} unique) in {num_chunks} chunks"
        )

        result = Parallel(n_jobs=self.parallelism)(
            delayed(self._get_affinities)(
                unique_smiles[chunk_idx * chunksize : (chunk_idx + 1) * chunksize],
            )
            for chunk_idx in tqdm(range(num_chunks))
        )
        unique_affinities = np.hstack(result)
        logging.info(f"Computed {num_unique_smiles} unique affinity values")

        # map the affinities back to the original library
        smiles_index = {s: i for i, s in enumerate(unique_smiles)}
        affinities = unique_affinities[[smiles_index[s] for s in query_library.smiles]]
        # FIXME ML-1190
        return affinities, unique_affinities


class SimilarityAffinityCalculator(AffinityCalculator, FingerprintAndSimilarityMixin):
    def __init__(
        self,
        reference_affinities: np.ndarray[float],
        reference_smiles: T.List[SMILES],
        sampler,
        kernel_params: KernelParameters,
        tversky_params: TverskyParameters,
        fp_generator_method: FingerprintGeneratorMethod,
        fp_generator_method_parameters: T.Union[
            MorganFingerPrintGeneratorParameters, None
        ],
        parallelism: int = 4,
        progress_bar: bool = False,
    ) -> None:
        super().__init__(progress_bar=progress_bar, parallelism=parallelism)

        self.reference_smiles = reference_smiles
        self.reference_affinities = reference_affinities

        self.kernel_params = kernel_params
        self.tversky_params = tversky_params
        self.sampler = sampler
        self.parallelism = parallelism

        self.fpf = FingerprintFeaturizer(
            sparse=False,
            sanitize=True,
            to_numpy=False,
            fp_generator_method=fp_generator_method,
            fp_generator_method_parameters=fp_generator_method_parameters,
        )

        self.reference_fingerprints = self.compute_fingerprints(self.reference_smiles)

    def gaussian_kernel(self, x: np.array):
        return np.exp(-1.0 / self.kernel_params.a**2 * (x - 1) ** 2)

    def _get_affinities(self, smiles: T.List[SMILES], baseline_affinities) -> np.array:
        query_fingerprints = self.compute_fingerprints(smiles)
        num_query_molecules = len(query_fingerprints)

        sim_mx = self.compute_tversky_similarity_mx(
            self.reference_fingerprints,
            query_fingerprints,
            a=self.tversky_params.a,
            b=self.tversky_params.b,
        )
        # apply the gaussian kernel
        sim_mx = self.gaussian_kernel(sim_mx)

        # append a row of 1-max to similarity matrix
        sim_mx = np.hstack([sim_mx, (1.0 - np.max(sim_mx, axis=1)).T.reshape(-1, 1)])

        sim_mx = np.exp(self.kernel_params.c * sim_mx) / np.sum(
            np.exp(self.kernel_params.c * sim_mx), axis=1, keepdims=True
        )

        #  get affinity matrix by replicating reference affinities and appending the baseline affinities as the last row
        affs = np.vstack(
            [
                np.tile(
                    self.reference_affinities,
                    (num_query_molecules, 1),
                ).T,
                baseline_affinities,
            ]
        )
        # affs is tiled to one column per query molecule, so np.matmul(sim_mx, affs) is
        # always square and only its diagonal is ever used -- computing the full
        # (num_query, num_query) product just to discard the off-diagonal entries costs
        # O(num_query^2) instead of O(num_query); sum the elementwise product instead.
        return np.sum(sim_mx * affs.T, axis=1)

    def get_affinities(
        self,
        query_library: Library,
        chunksize: int = 8192,
    ) -> np.array:
        library_size = len(query_library.nsynthon_id)

        logging.info(
            f"Computing baseline affinities for {library_size} smiles using {self.sampler.__class__.__name__}",
        )

        unique_smiles = sorted(set(query_library.smiles))
        num_unique_smiles = len(unique_smiles)

        _, unique_baseline_affinities = RandomAffinityCalculator(
            sampler=self.sampler, parallelism=self.parallelism
        ).get_affinities(query_library=query_library, chunksize=chunksize)

        affinities = np.zeros(library_size)

        num_chunks = (num_unique_smiles - 1) // chunksize + 1

        logging.info(
            f"Computing affinities for {library_size} smiles({num_unique_smiles} unique) in {num_chunks} chunks"
        )

        result = Parallel(n_jobs=self.parallelism)(
            delayed(self._get_affinities)(
                unique_smiles[chunk_idx * chunksize : (chunk_idx + 1) * chunksize],
                unique_baseline_affinities[
                    chunk_idx * chunksize : (chunk_idx + 1) * chunksize
                ],
            )
            for chunk_idx in tqdm(range(num_chunks))
        )
        unique_affinities = np.hstack(
            result
        )  # fixme, this would be better if this were re-implemented as a streaming operation to limit OOMs

        logging.info(f"Computed {num_unique_smiles} unique affinity values")

        # map the affinities back to the original library
        smiles_index = {s: i for i, s in enumerate(unique_smiles)}
        affinities = unique_affinities[[smiles_index[s] for s in query_library.smiles]]

        return affinities, unique_affinities
