import logging
import queue
from collections import Counter
from multiprocessing import Process

import pytest
from del_simulator.library_generator import (
    LibraryGenerator,
)

import json
import pandas as pd
from pathlib import Path
from del_simulator.core import (
    ReactionScheme,
    BuildingBlocksSmiles,
    CollectedSynthonOutput,
)

from pandas.testing import assert_frame_equal


@pytest.fixture(scope="module")
def simple_data():
    with open(
        Path(__file__).parent / "resources/dag0.json",
        "r",
    ) as jsonfile:
        reaction_scheme = ReactionScheme.model_validate(json.load(jsonfile))

    with open(
        Path(__file__).parent / "resources/bbs0.json",
        "r",
    ) as jsonfile:
        bbs = BuildingBlocksSmiles.model_validate(json.load(jsonfile))

    lib = pd.read_csv(Path(__file__).parent / "resources/dag0_bbs0_output.csv")

    return bbs, reaction_scheme, lib


@pytest.fixture(scope="module")
def deprotection_data():
    with open(
        Path(__file__).parent / "resources/dag1.json",
        "r",
    ) as jsonfile:
        reaction_scheme = ReactionScheme.model_validate(json.load(jsonfile))

    with open(
        Path(__file__).parent / "resources/bbs1.json",
        "r",
    ) as jsonfile:
        bbs = BuildingBlocksSmiles.model_validate(json.load(jsonfile))

    lib = pd.read_csv(Path(__file__).parent / "resources/dag1_bbs1_output.csv")

    return bbs, reaction_scheme, lib


@pytest.fixture(scope="module")
def branched_graph():
    with open(
        Path(__file__).parent / "resources/branched_dag.json",
        "r",
    ) as jsonfile:
        reaction_scheme = ReactionScheme.model_validate(json.load(jsonfile))

    with open(
        Path(__file__).parent / "resources/branched_bbs.json",
        "r",
    ) as jsonfile:
        bbs = BuildingBlocksSmiles.model_validate(json.load(jsonfile))

    lib = pd.read_csv(Path(__file__).parent / "resources/branched_output.csv")

    return bbs, reaction_scheme, lib


@pytest.mark.parametrize(
    "test_data", ["simple_data", "deprotection_data", "branched_graph"]
)
# this request / fixture name business is so we can parametrize across fixtures
# https://github.com/pytest-dev/pytest/issues/349
def test_library_generator(test_data, request, tmpdir):
    test_bbs, test_dag, ref_lib = request.getfixturevalue(test_data)

    library_generator = LibraryGenerator(
        raw_bbs=test_bbs,
        reaction_scheme=test_dag,
        num_workers=4,
        output_path=str(tmpdir),
        min_abundance_to_output=0,
    )

    library_generator.generate()
    test_lib = pd.read_csv(tmpdir / "library.csv")

    # do compare dataframes, need to make sure they are sorted and rounded
    for lib in [test_lib, ref_lib]:
        lib.sort_values(by=["nsynthon_id", "smiles", "relative_fraction"], inplace=True)
        lib.reset_index(drop=True, inplace=True)

    assert_frame_equal(ref_lib, test_lib)


def test_building_block_subsets_default_is_not_shared_between_instances(
    simple_data, tmpdir
):
    """
    building_block_subsets used to default to a mutable {} literal, so every
    LibraryGenerator constructed without this argument shared the exact same dict object.
    Not actively mutated in place anywhere today, but a footgun: mutating one instance's
    building_block_subsets must not be visible on another instance's.
    """
    test_bbs, test_dag, _ = simple_data

    gen1 = LibraryGenerator(
        raw_bbs=test_bbs,
        reaction_scheme=test_dag,
        num_workers=1,
        output_path=str(tmpdir.mkdir("gen1")),
    )
    gen2 = LibraryGenerator(
        raw_bbs=test_bbs,
        reaction_scheme=test_dag,
        num_workers=1,
        output_path=str(tmpdir.mkdir("gen2")),
    )

    assert gen1.building_block_subsets is not gen2.building_block_subsets

    gen1.building_block_subsets["B0"] = (0, 1)
    assert gen2.building_block_subsets == {}


def test_validate_raises_on_invalid_building_block_when_fail_on_invalid_bbs(
    simple_data, tmpdir
):
    """
    _validate checks that every building block at an edge's target node actually matches
    the reaction template's second-reactant SMARTS. "CCCC" (butane) has no carboxylic acid
    group, so it can't react in the B0->B1 Amidation step -- this should be flagged when
    fail_on_invalid_bbs=True, which was previously untested.
    """
    test_bbs, test_dag, _ = simple_data
    broken_bbs = BuildingBlocksSmiles(root=dict(test_bbs.root))
    broken_bbs["B1"] = [*test_bbs["B1"], "CCCC"]

    with pytest.raises(ValueError, match="Invalid building block"):
        LibraryGenerator(
            raw_bbs=broken_bbs,
            reaction_scheme=test_dag,
            num_workers=1,
            output_path=str(tmpdir),
            fail_on_invalid_bbs=True,
        )


def test_log_edge_outcome_summary_warns_on_high_zero_product_rate(
    simple_data, tmpdir, caplog
):
    test_bbs, test_dag, _ = simple_data
    generator = LibraryGenerator(
        raw_bbs=test_bbs,
        reaction_scheme=test_dag,
        num_workers=1,
        output_path=str(tmpdir),
    )
    edge = ("B0", "B1")

    with caplog.at_level(logging.WARNING):
        generator._log_edge_outcome_summary(
            edge_attempts=Counter({edge: 25}),
            edge_zero_product=Counter({edge: 24}),
            edge_skipped=Counter(),
        )

    assert any(
        "reactant order" in record.message for record in caplog.records
    ), "expected a warning about a high zero-product rate for the edge"


def test_save_output_skips_products_below_min_abundance(simple_data, tmpdir):
    test_bbs, test_dag, _ = simple_data
    generator = LibraryGenerator(
        raw_bbs=test_bbs,
        reaction_scheme=test_dag,
        num_workers=1,
        output_path=str(tmpdir),
    )

    q = queue.Queue()
    q.put(
        CollectedSynthonOutput(
            nsynthon_id="0_0",
            smiles_and_abundances=[("CCO", 0.5), ("CCN", 1e-10)],
        )
    )
    q.put({})

    generator._save_output(q, num_synthons=1, min_abundance=1e-5)

    written = pd.read_csv(f"{generator.output_path}/library.csv")
    assert set(written["smiles"]) == {"CCO"}


class _JoinSpyProcess(Process):
    """
    Module-level (not a local class inside the test) so it stays picklable under
    spawn-based multiprocessing (macOS/Windows) -- pickling a Process instance requires
    pickling a reference to its class by module + qualname, which fails for a class
    defined inside a function.
    """

    joined = []

    def join(self, *args, **kwargs):
        result = super().join(*args, **kwargs)
        type(self).joined.append(self)
        return result


def test_generate_joins_the_consumer_process(simple_data, tmpdir, monkeypatch):
    """
    generate() started the consumer process (draining the queue and writing library.csv)
    but never joined it before exiting the `with Manager()` block, which shuts down the
    manager server -- a race that can raise BrokenPipeError/EOFError in the consumer or
    truncate library.csv if the consumer hasn't finished by then. Verify the consumer is
    actually joined (and therefore finished) before generate() returns.
    """
    import del_simulator.library_generator as library_generator_module

    test_bbs, test_dag, _ = simple_data
    generator = LibraryGenerator(
        raw_bbs=test_bbs,
        reaction_scheme=test_dag,
        num_workers=1,
        output_path=str(tmpdir),
        min_abundance_to_output=0,
    )

    _JoinSpyProcess.joined = []
    monkeypatch.setattr(library_generator_module, "Process", _JoinSpyProcess)

    generator.generate()

    assert len(_JoinSpyProcess.joined) == 1
    assert not _JoinSpyProcess.joined[0].is_alive()
