import random
from omegaconf import DictConfig, OmegaConf
from rdkit import Chem
import typing as T
from dataclasses import field, dataclass
import rdkit
from rdkit.Chem import rdChemReactions
from enum import Enum

import del_simulator.yield_generator as yield_generator
from pydantic import BaseModel, computed_field, RootModel

import numpy as np

import numpy as np

SMILES = T.NewType("SMILES", str)
SMARTSTemplate = T.NewType("SMARTSTemplate", str)
STEP_NAME = T.NewType("STEP_NAME", str)
NSYNTHON_ID = T.NewType("NSYNTHON_ID", str)

BLANK_BUILDING_BLOCK_SMILES = "Null"
SKIPPED_NSYTHON_BIT_ENCODING = "X"
AVOGADROS_CONSTANT = 6.022 * 10**23


class ReactionSchemeEdgeAttributes(BaseModel):
    # FIXME #870
    # the reaction template is a smarts-- this is NOT an rdkit reaction object
    # because downstream we utilize the reaction utils to symmetrize the reaction

    reaction_name: str
    reaction_template: SMARTSTemplate
    yield_generator_class: str
    yield_generator_params: T.Dict[str, T.Any]
    source: STEP_NAME
    target: STEP_NAME

    class Config:
        arbitrary_types_allowed = True

    @computed_field
    @property
    def yield_generator(self) -> yield_generator.ReactionYieldsGenerator:
        rxn_yield_generator = getattr(yield_generator, self.yield_generator_class)(
            **self.yield_generator_params
        )
        return yield_generator.ReactionYieldsGenerator(rxn_yield_generator)

    @computed_field
    @property
    def reaction(self) -> rdChemReactions.ChemicalReaction:
        return rdChemReactions.ReactionFromSmarts(self.reaction_template)


class ReactionScheme(BaseModel):
    directed: T.Literal[True]
    multigraph: T.Literal[False]
    graph: T.Dict

    nodes: T.List[T.Dict[str, STEP_NAME]]
    links: T.List[ReactionSchemeEdgeAttributes]


class BuildingBlocksMols(RootModel):
    root: T.Dict[STEP_NAME, T.List[Chem.Mol]]

    class Config:
        arbitrary_types_allowed = True

    def __getitem__(self, key):
        return self.root[key]

    def __setitem__(self, key, value):
        self.root[key] = value

    def __delitem__(self, key):
        del self.root[key]

    def __getattr__(self, item):
        return getattr(self.root, item)


class BuildingBlocksSmiles(RootModel):
    root: T.Dict[STEP_NAME, T.List[SMILES]]

    def __getitem__(self, key):
        return self.root[key]

    def __setitem__(self, key, value):
        self.root[key] = value

    def __delitem__(self, key):
        del self.root[key]

    def __getattr__(self, item):
        return getattr(self.root, item)


class FingerprintGeneratorMethod(Enum):
    morgan = 0  # AllChem.GetMorganFingerprintAsBitVect()
    rdkit = 1  # Chem.RDKFingerprint()
    maccs = 2  # MACCSkeys.GenMACCSKeys(x)


@dataclass
class MorganFingerPrintGeneratorParameters:
    radius: int
    nBits: int
    # useFeatures: bool


@dataclass
class RDKitFingerPrintGeneratorParameters:
    # named to match Chem.RDKFingerprint's actual maxPath kwarg -- "radius" (copied from
    # MorganFingerPrintGeneratorParameters) doesn't exist on RDKFingerprint's signature at
    # all, so any use of the "rdkit" fp_generator_method crashed immediately
    maxPath: int


@dataclass
class FingerPrintFeaturizerConfig:
    fp_generator_method: str  # todo add rdkit, ecfp
    # T.Any, not a Union of the concrete dataclasses -- OmegaConf refuses to merge a Union of
    # >1 container type ("Unions of containers are not supported") once this class sits inside
    # a List (FeaturizerConfig.featurizer_parameters via FeaturizerRunsConfig.featurizer_runs),
    # which made featurize_dataset.py's config loading crash unconditionally. __post_init__
    # below already does its own isinstance-based promotion, so the static type isn't load-bearing.
    fp_generator_method_parameters: T.Any
    parallelism: int = 4
    chunksize: int = 16384
    smiles_column_name: str = "smiles"

    def __post_init__(self):
        params = self.fp_generator_method_parameters

        if isinstance(
            params,
            (MorganFingerPrintGeneratorParameters, RDKitFingerPrintGeneratorParameters),
        ):
            return

        if isinstance(params, DictConfig):
            params = OmegaConf.to_container(params, resolve=True)

        if params is None:
            params = {}

        if self.fp_generator_method == "morgan":
            self.fp_generator_method_parameters = MorganFingerPrintGeneratorParameters(
                **params
            )
        elif self.fp_generator_method == "rdkit":
            self.fp_generator_method_parameters = RDKitFingerPrintGeneratorParameters(
                **params
            )
        elif self.fp_generator_method == "maccs":
            self.fp_generator_method_parameters = None
        else:
            raise ValueError(f"Unknown fp_generator_method: {self.fp_generator_method}")


@dataclass
class FeaturizerConfig:
    input_path: str
    output_path: str
    featurizer_method: str
    featurizer_parameters: FingerPrintFeaturizerConfig  # FIXME union does not work -- rplace with post_init_


@dataclass
class FeaturizerRunsConfig:
    featurizer_runs: T.List[FeaturizerConfig]


@dataclass
class Library:
    nsynthon_id: T.List[str]
    smiles: T.List[str]
    relative_fraction: np.array = None


@dataclass
class LibraryWithAffinities(Library):
    pKd: T.Dict[str, np.ndarray] = field(default_factory=lambda: {})
    Kd: T.Dict[str, np.ndarray] = field(default_factory=lambda: {})

    def __post_init__(self):
        if not self.Kd and self.pKd:
            self.Kd = {key: 10 ** (-1.0 * val) for key, val in self.pKd.items()}

        if not self.pKd and self.Kd:
            self.pKd = {key: -np.log10(val) for key, val in self.Kd.items()}


@dataclass
class NSynthonComponents:
    nsynthon_id: NSYNTHON_ID
    building_blocks: T.Dict[STEP_NAME, Chem.Mol]


@dataclass
class CollectedSynthonOutput:
    nsynthon_id: NSYNTHON_ID
    smiles_and_abundances: T.List[T.Tuple[SMILES, float]]
    # per-edge counts of {"reacted": n, "zero_product": n, "skipped_no_bb": n} for this nsynthon,
    # used to track reaction-template failure rates across a full library generation run
    edge_outcome_counts: T.Dict[T.Tuple[STEP_NAME, STEP_NAME], T.Dict[str, int]] = (
        field(default_factory=dict)
    )


@dataclass(frozen=True)
class ReactionSchemeNodeData:
    bb_to_add: Chem.Mol = field(hash=False, default=None)
    products: T.List[Chem.Mol] = field(hash=False, default=None)
    relative_product_fractions: T.List[float] = field(hash=False, default=None)


@dataclass
class SelectionExperimentResults:
    smiles: list[str]
    nsynthon_id: T.List[str]
    concentration: np.array


@dataclass
class ReadoutExperimentResults:
    nsynthon_id: T.List[str]
    count: np.array


@dataclass
class LibraryValidationConfig:
    building_block_path: str
    reaction_scheme_path: str
    validation_schema: T.Dict[str, int]


@dataclass
class LibraryGenerationConfig:
    building_block_path: str
    reaction_scheme_path: str
    output_path: str
    building_block_subsets: T.Optional[T.Dict[str, T.List[int]]] = field(
        default_factory=lambda: {}
    )
    num_cpu: int = 1
    chunksize: int = 100
    return_all_products: bool = True


# AFFINITY GENERATION


@dataclass
class TruncatedNormalParameters:
    loc: float
    scale: float
    truncated_min: float = 0.0
    truncated_max: float = 9.0  # nanomolar
    seed: int = 42


@dataclass
class LogNormalParameters:
    s: float
    loc: float
    scale: float
    seed: int = 42


@dataclass
class SigmoidalParameters:
    a: float
    x0: float


@dataclass
class TverskyParameters:
    a: float
    b: float


@dataclass
class KernelParameters:
    a: float
    c: float


class Distribution(str, Enum):
    truncated_normal = "truncated_normal"
    lognormal = "lognormal"


class AffinityMethod(str, Enum):
    similarity = "similarity"
    random = "random"


@dataclass
class AffinityGenSimilarityMethodParameters:
    reference_affinity_path: str
    sampler: Distribution  # fixme poor naming
    kernel_params: KernelParameters
    tversky_params: TverskyParameters
    fp_generator_method: str
    fp_generator_method_parameters: MorganFingerPrintGeneratorParameters
    reference_affinity_field_name: str = "pKd"
    reference_smiles_field_name: str = "smiles"
    chunksize: int = 16384
    parallelism: int = 4
    progress_bar: bool = False
    sampler_params: T.Any = None

    def __post_init__(self):
        unvalidated_payload = self.sampler_params or {}

        if isinstance(self.sampler, str):
            self.sampler = Distribution(self.sampler)

        if isinstance(self.kernel_params, T.Dict):
            self.kernel_params = KernelParameters(**self.kernel_params)

        if isinstance(self.tversky_params, T.Dict):
            self.tversky_params = TverskyParameters(**self.tversky_params)

        if isinstance(self.fp_generator_method_parameters, T.Dict):
            self.fp_generator_method_parameters = MorganFingerPrintGeneratorParameters(
                **self.fp_generator_method_parameters
            )

        if isinstance(
            unvalidated_payload,
            (TruncatedNormalParameters, LogNormalParameters),
        ):
            return

        if self.sampler is Distribution.truncated_normal:
            self.sampler_params = TruncatedNormalParameters(**unvalidated_payload)
        elif self.sampler is Distribution.lognormal:
            self.sampler_params = LogNormalParameters(**unvalidated_payload)
        else:
            raise ValueError(f"Unknown sampler method {self.sampler!r}")


@dataclass
class AffinityGenRandomMethodParameters:  # FIXME code duplucation with above mehotd.
    sampler: Distribution
    chunksize: int = 16384
    parallelism: int = 4
    progress_bar: bool = False
    sampler_params: T.Any = None

    def __post_init__(self):
        unvalidated_payload = self.sampler_params or {}

        if isinstance(self.sampler, str):
            self.sampler = Distribution(self.sampler)

        if isinstance(
            unvalidated_payload,
            (TruncatedNormalParameters, LogNormalParameters),
        ):
            return
        if self.sampler is Distribution.truncated_normal:
            self.sampler_params = TruncatedNormalParameters(**unvalidated_payload)
        elif self.sampler is Distribution.lognormal:
            self.sampler_params = LogNormalParameters(**unvalidated_payload)
        else:
            raise ValueError(f"Unknown sampler method {self.sampler!r}")


@dataclass
class AffinityGenRunConfig:
    method: AffinityMethod
    method_parameters: T.Any = None

    def __post_init__(self):
        unvalidated_payload = self.method_parameters or {}

        if isinstance(
            unvalidated_payload,
            (AffinityGenRandomMethodParameters, AffinityGenSimilarityMethodParameters),
        ):
            return
        if self.method is AffinityMethod.similarity:
            self.method_parameters = AffinityGenSimilarityMethodParameters(
                **unvalidated_payload
            )
        elif self.method is AffinityMethod.random:
            self.method_parameters = AffinityGenRandomMethodParameters(
                **unvalidated_payload
            )
        else:
            raise ValueError(f"Unknown affinity method {self.method!r}")


@dataclass
class AffinityGenConfig:
    input_library_path: str
    output_path: str
    runs: T.Dict[str, AffinityGenRunConfig]
    num_query_molecules: T.Optional[int] = None
    input_library_sep: str = ","


# SELECTION AND READOUT


@dataclass
class SingleSelectionExperimentParameters:
    binding_sites: T.List[str]
    recovery_fractions: T.List[float]
    binding_site_concentrations_M: T.List[float]
    num_selection_rounds: int


@dataclass
class SelectionExperimentConfig:
    input_library_path: str
    input_affinity_paths: T.Dict[str, str]
    initial_library_amount_mol: float
    experiment_volume_L: float
    selection_experiments: T.Dict[str, SingleSelectionExperimentParameters]
    num_query_molecules: T.Optional[int] = None
    output_path: T.Optional[str] = None


@dataclass
class PCRGaussianEfficiencyParameters(TruncatedNormalParameters):
    seed: int = 42


@dataclass
class AmplificationRunConfig:
    num_reads: int
    output_path: str
    num_pcr_amplification_cycles: int = 0
    pcr_efficiency_parameters: T.Optional[PCRGaussianEfficiencyParameters] = None


@dataclass
class AmplificationReadoutConfig:
    output_path_prefix: str
    readout_seed: int
    readout_runs: T.Dict[str, AmplificationRunConfig]


# ENRICHMENT CALCULATIONS


@dataclass
class ZScoreProcessorParameters:
    library_size: int
    total_number_of_reads: int
    unique_mols_observed: int


@dataclass
class RatioTestProcessorParameters:
    enrichment_threshold: float
    total_number_of_reads_in_sample_1: int
    total_number_of_reads_in_sample_2: int
    library_size: int


@dataclass
class EnrichmentCalcConfig:
    method: str
    method_params: ZScoreProcessorParameters | RatioTestProcessorParameters
    input_columns: T.List[str]
    output_column: str


# DATA PREP
@dataclass
class DataPrepRunConfig:
    intended_product_path: str
    target_data_path: str
    ntc_data_path: str
    output_path: str
    fp_generator_method: str
    fp_generator_method_parameters: MorganFingerPrintGeneratorParameters  # FIXME union does not work -- rplace with post_init_
    bbs_per_nsynthon: int = (
        3  # how many building blocks define a single nsynthon # fixme should be able to get this from the library...
    )
    num_bbs_to_aggregate: int = (
        2  # numver of building blocks to aggregate, e.g. 2 for disynthon, 3 for trisynthon
    )


@dataclass
class DataPrepConfig:
    runs: T.Dict[str, DataPrepRunConfig]


# Training and Inference
@dataclass
class InferenceConfig:
    # FIXME currently RF reads the fingerprint path# and the chemprop reads and refeaturizes the smiles
    # ideally, we want both to either be able to featurize or to read the data if it exists
    smiles_path: T.Optional[str] = None
    featurized_data_path: T.Optional[str] = None
    affinity_path: T.Optional[str] = None
    affinity_threshold: T.Optional[float] = None
    affinity_column: T.Optional[str] = "pKd"
    sample_size: T.Optional[int] = None
    output_predictions_path: T.Optional[str] = None
    # separator for smiles_path -- only read by DELSimulatorChemPropModel, which parses
    # smiles_path as a real CSV (RF reads pre-featurized fingerprints instead)
    separator: str = ","


class EnrichmentMethod(str, Enum):
    scaled_poisson_ratio_test = "scaled_poisson_ratio_test"
    normalized_zscore = "normalized_zscore"
    count_ratio = "count_ratio"


class MLMethod(str, Enum):
    random_forest = "random_forest"
    chemprop = "chemprop"


@dataclass
class RandomForestClassifierParameters:
    n_estimators: int = 100
    max_depth: int = 20
    min_samples_split: int = 2
    min_samples_leaf: int = 1
    max_features: T.Literal["sqrt", "log2", None] = "sqrt"


@dataclass
class ChemPropGNNParameters:
    batch_size: int = 128
    num_workers: int = 8
    random_seed: int = 42
    max_epochs: int = 30


@dataclass
class MLConfig:
    method: MLMethod = MLMethod.random_forest
    random_seed: int = 42
    method_parameters: T.Any = None

    def __post_init__(self):
        unvalidated_payload = self.method_parameters or {}

        if isinstance(
            unvalidated_payload,
            (RandomForestClassifierParameters, ChemPropGNNParameters),
        ):
            return
        if self.method is MLMethod.random_forest:
            self.method_parameters = RandomForestClassifierParameters(
                **unvalidated_payload
            )
        elif self.method is MLMethod.chemprop:
            self.method_parameters = ChemPropGNNParameters(**unvalidated_payload)
        else:
            raise ValueError(f"Unknown method {self.method!r}")


@dataclass
class ScaledPoissonRatioTestMethodParameters:
    library_diversity: int
    poisson_ratio: float


@dataclass
class NormalizedZScoreMethodParameters:
    library_diversity: int


@dataclass
class TrainingAndInferenceConfig:
    input_data_path: str
    model_output_path: str
    metrics_output_path: str
    ml_config: MLConfig  # fixme add defaults
    enrichment_method: EnrichmentMethod = EnrichmentMethod.scaled_poisson_ratio_test
    enrichment_method_params: T.Optional[T.Dict[str, T.Any]] = None
    enrichment_threshold: float = 5
    n_cv_splits: int = 5
    inference: T.Optional[InferenceConfig] = None

    def __post_init__(self):
        unvalidated_payload = self.enrichment_method_params or {}

        if isinstance(
            unvalidated_payload,
            (ScaledPoissonRatioTestMethodParameters, NormalizedZScoreMethodParameters),
        ):
            return

        if self.enrichment_method is EnrichmentMethod.scaled_poisson_ratio_test:
            self.enrichment_method_params = ScaledPoissonRatioTestMethodParameters(
                **unvalidated_payload
            )
        elif self.enrichment_method is EnrichmentMethod.normalized_zscore:
            self.enrichment_method_params = NormalizedZScoreMethodParameters(
                **unvalidated_payload
            )
        elif self.enrichment_method is EnrichmentMethod.count_ratio:
            return  # count_ratio never reads enrichment_method_params
        else:
            raise ValueError(f"Unknown enrichment method {self.enrichment_method!r}")

TrainingAndInferenceRunConfig = TrainingAndInferenceConfig

@dataclass
class TrainingAndInferenceConfig:
    runs: T.Dict[str, TrainingAndInferenceRunConfig]
