from collections import defaultdict, Counter
import csv
import itertools
from pathlib import Path
import networkx as nx


import typing as T
import logging

from multiprocessing import Process, Pool, Manager
from rdkit import Chem
import time
from tqdm import tqdm
import json

from del_simulator.utils.utils import setup_logger, clean_building_blocks
import typing as T
from del_simulator.reaction_graph import ReactionGraph
from del_simulator.core import (
    ReactionScheme,
    BuildingBlocksSmiles,
    BuildingBlocksMols,
    NSynthonComponents,
    BLANK_BUILDING_BLOCK_SMILES,
    SKIPPED_NSYTHON_BIT_ENCODING,
)

# need to set the default pickle properties for the rdkit objects so that smiles get pickled


class LibraryGenerator:
    def __init__(
        self,
        reaction_scheme: ReactionScheme,
        raw_bbs: BuildingBlocksSmiles,
        num_workers: int = 4,
        chunksize: int = 100,
        min_abundance_to_output: float = 1e-9,
        building_block_subsets: T.Optional[T.Dict[str, T.Tuple[int, int]]] = None,
        output_path: str = "output",
        fail_on_invalid_bbs: bool = False,
        return_all_products: bool = True,
        edge_failure_rate_warn_threshold: float = 0.95,
        edge_failure_min_attempts: int = 20,
    ):
        self.output_path = f"{output_path}/"

        # Need to set this s.t. properties get pickled https://github.com/rdkit/rdkit/issues/2470
        Chem.SetDefaultPickleProperties(Chem.PropertyPickleOptions.AllProps)

        Path(f"{self.output_path}").mkdir(parents=True, exist_ok=True)

        self.logger = setup_logger(
            "LibraryGenerator", f"{self.output_path}/library_generator.log"
        )

        self.reaction_graph = ReactionGraph(reaction_scheme)
        self.nsynthon_ordering = list(nx.dfs_preorder_nodes(self.reaction_graph.graph))

        # write out the nsynthon ordering
        with open(f"{self.output_path}/nsynthon_ordering.json", "w") as f:
            json.dump(self.nsynthon_ordering, f)

        self.building_block_subsets = building_block_subsets or {}

        self.clean_bbs = clean_building_blocks(
            raw_bbs
        )  # all tasks do a full clean of the bbs

        self._validate(fail_on_invalid_bbs)

        self.num_workers = num_workers
        self.chunksize = chunksize

        self.min_abundance_to_output = min_abundance_to_output
        self.return_all_products = return_all_products

        # Thresholds for flagging a reaction-scheme edge whose reactant order likely doesn't
        # match how it's invoked (see _log_edge_outcome_summary): an edge is only warned about
        # once it has been attempted at least edge_failure_min_attempts times (across distinct
        # building blocks) and still fails to produce any product edge_failure_rate_warn_threshold
        # of the time.
        self.edge_failure_rate_warn_threshold = edge_failure_rate_warn_threshold
        self.edge_failure_min_attempts = edge_failure_min_attempts

    def _validate(self, fail_on_invalid_bbs: bool = False):
        self.logger.info("Validating the building blocks")
        valid = True
        # validate that the second reactant in the reaction scheme actually can react with the smarts definition
        invalid_bbs = defaultdict(list)
        for edge in self.reaction_graph.graph.edges():
            reactionSmarts = nx.get_edge_attributes(
                self.reaction_graph.graph, "reaction_template"
            )[edge]

            try:
                second_reactant_smarts = reactionSmarts.split(">>")[0].split(".")[1]
                for bb_idx, bb in enumerate(self.clean_bbs[edge[1]]):
                    if bb is not None:
                        matches = bb.GetSubstructMatches(
                            Chem.MolFromSmarts(second_reactant_smarts)
                        )
                        if len(matches) == 0:
                            invalid_bbs[edge[1]].append((bb_idx, bb))
                            valid = False

            except IndexError:
                pass  #  a single component reaciton, nothing to do

        if not (valid):
            for step in invalid_bbs:
                self.logger.error(
                    f"The following building blocks will not generate products when used in {step}: {', '.join([f'{idx} : {Chem.MolToSmiles(Chem.RemoveHs(mol))}' for idx,mol in invalid_bbs[step]])}"
                )

            if fail_on_invalid_bbs:
                raise ValueError(
                    "Invalid building block / reaction template detected in the reaction scheme"
                )

    def _construct_all_nsynthon_definitions(self) -> T.List[NSynthonComponents]:
        """
        Constructs all possible combinations of building blocks.

        and returns 1/num_tasks a portion of them (starting at task_idx,)

        Returns:
            A list of NSynthonBuildingBlocks objects representing
            all possible combinations of building blocks.

        """

        logging.info(
            f"Constructing All N-Synthons; \n Subsetting the building blocks: \n {self.building_block_subsets}"
        )
        self.logger.info(
            f"The nsynthon bits are encoded in the following order: {self.nsynthon_ordering}"
        )

        # construct all possible root->leaf traversals of the reaction scheme
        all_paths = nx.all_simple_paths(
            self.reaction_graph.graph,
            self.reaction_graph.root_node_id,
            self.reaction_graph.leaf_nodes,
            cutoff=None,
        )

        all_nsynthons = []

        self.logger.info("Generating all possible NSynthons for the reaction scheme...")

        for path in all_paths:
            all_bb_indices = itertools.product(
                *[
                    (
                        range(
                            self.building_block_subsets.get(
                                step, [0, len(self.clean_bbs[step])]
                            )[0],
                            self.building_block_subsets.get(
                                step, [0, len(self.clean_bbs[step])]
                            )[1],
                        )
                        if step in path
                        else [SKIPPED_NSYTHON_BIT_ENCODING]
                    )
                    for step in self.nsynthon_ordering
                ]
            )

            for bb_indices in tqdm(all_bb_indices):
                all_nsynthons.append(
                    NSynthonComponents(
                        nsynthon_id="_".join([str(x) for x in bb_indices]),
                        # the building block dictonary is ordered by step name, which is ordered by its order
                        # in the reaction traversal path (keys maintain insertion order!)
                        # this is how we encode the specific path to be traversed
                        building_blocks={
                            step_name: self.clean_bbs[step_name][
                                bb_indices[self.nsynthon_ordering.index(step_name)]
                            ]
                            for step_name in path
                        },
                    )
                )

        return all_nsynthons

    def _log_edge_outcome_summary(
        self,
        edge_attempts: T.Dict[T.Tuple[str, str], int],
        edge_zero_product: T.Dict[T.Tuple[str, str], int],
        edge_skipped: T.Dict[T.Tuple[str, str], int],
    ) -> None:
        """
        Log per-edge reaction outcome counts accumulated over the whole generation run, and warn
        on edges whose zero-product rate suggests the reaction template's reactant order doesn't
        match how it's invoked (RunReactants is order-sensitive and is not automatically
        symmetrized -- see reaction_graph.py), rather than expected truncated/non-reactive
        building blocks.
        """
        all_edges = set(edge_attempts) | set(edge_skipped)
        self.logger.info("Per-edge reaction outcome summary:")

        for edge in sorted(all_edges, key=str):
            attempts = edge_attempts.get(edge, 0)
            zero_product = edge_zero_product.get(edge, 0)
            skipped = edge_skipped.get(edge, 0)
            zero_product_rate = zero_product / attempts if attempts else 0.0

            self.logger.info(
                "  %s: %s attempted (with a building block supplied), %s produced zero products "
                "(%.1f%%), %s skipped (no building block supplied)",
                edge,
                attempts,
                zero_product,
                100 * zero_product_rate,
                skipped,
            )

            if (
                attempts >= self.edge_failure_min_attempts
                and zero_product_rate >= self.edge_failure_rate_warn_threshold
            ):
                self.logger.warning(
                    "Edge %s produced zero products in %.1f%% of %s attempts across distinct "
                    "building blocks. If this edge isn't expected to be almost entirely "
                    "unreactive, this may mean the reaction template's reactant order doesn't "
                    "match how it's invoked here, rather than a truncated/non-reactive building "
                    "block -- check the reaction_scheme's template for this edge.",
                    edge,
                    100 * zero_product_rate,
                    attempts,
                )

    def _save_output(
        self, queue, num_synthons: int, min_abundance: float = 1e-5
    ) -> None:
        self.logger.info(f"Saving library to {self.output_path}/library.csv")
        self.logger.info(
            "Minimum abundance of product to be saved : {}".format(min_abundance)
        )

        csvfile = open(f"{self.output_path}/library.csv", "w", newline="")

        writer = csv.writer(csvfile)
        writer.writerow(["nsynthon_id", "smiles", "relative_fraction"])
        pbar = tqdm(total=num_synthons, mininterval=10.0)

        edge_attempts = Counter()
        edge_zero_product = Counter()
        edge_skipped = Counter()

        while True:
            nsynthon_products = queue.get()
            pbar.update()

            if nsynthon_products == {}:
                self._log_edge_outcome_summary(
                    edge_attempts, edge_zero_product, edge_skipped
                )
                self.logger.info("Library Generation Complete!")
                csvfile.close()
                break

            for edge, counts in nsynthon_products.edge_outcome_counts.items():
                edge_attempts[edge] += counts.get("reacted", 0) + counts.get(
                    "zero_product", 0
                )
                edge_zero_product[edge] += counts.get("zero_product", 0)
                edge_skipped[edge] += counts.get("skipped_no_bb", 0)

            for smiles, abundance in nsynthon_products.smiles_and_abundances:
                if abundance < min_abundance:
                    continue
                writer.writerow(
                    [
                        nsynthon_products.nsynthon_id,
                        smiles,
                        abundance,
                    ]
                )

    def _generate_into_queue(self, arg) -> None:
        (
            queue,
            nsython_components,
        ) = arg

        nsynthon_products = self.reaction_graph.execute_reaction_graph(
            nsython_components, self.return_all_products
        )

        queue.put(nsynthon_products)

    def generate(
        self,
    ) -> None:
        start_time = time.time()

        all_nsynthons = self._construct_all_nsynthon_definitions()

        num_synthons = len(all_nsynthons)

        self.logger.info(
            "LibraryGenerator will generate %s n-synthons using workers: %s; chunksize %s;",
            num_synthons,
            self.num_workers,
            self.chunksize,
        )
        with Manager() as manager:
            queue = manager.Queue()

            consumer = Process(
                target=self._save_output,
                args=(queue, num_synthons, self.min_abundance_to_output),
            )
            consumer.start()

            worker_pool = Pool(
                processes=self.num_workers,
            )

            args = ((queue, nsynthon) for nsynthon in all_nsynthons)

            # _generate_into_queue returns None (it writes to `queue`); drain the iterator
            # to drive execution without collecting the (unused) results or materializing
            # a second copy of all_nsynthons via a list.
            for _ in worker_pool.imap(
                self._generate_into_queue, args, chunksize=self.chunksize
            ):
                pass

            queue.put({})

            worker_pool.close()
            worker_pool.join()
            # Without this, exiting the `with Manager()` block below can shut down the
            # manager server before the consumer finishes draining the queue and closing
            # library.csv, causing intermittent errors/truncated output.
            consumer.join()

        self.logger.info(
            "Finished generating %s-member library in %s s. Output saved to %s",
            num_synthons,
            round(time.time() - start_time, 2),
            self.output_path,
        )
