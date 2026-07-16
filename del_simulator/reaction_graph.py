import networkx as nx
import typing as T
import logging
from collections import Counter
from functools import lru_cache
from itertools import pairwise
import math
from dataclasses import replace

from rdkit import Chem, RDLogger
from rdkit.Chem import rdChemReactions
from rdkit import Chem
from rdkit.Chem import AllChem, Mol
from rdkit.Chem.rdChemReactions import ChemicalReaction
from rdkit.Chem import rdmolops


from del_simulator.core import (
    SMARTSTemplate,
    SMILES,
    STEP_NAME,
    ReactionScheme,
    ReactionSchemeNodeData,
    CollectedSynthonOutput,
    NSynthonComponents,
)


from typing import (
    List,
    Sequence,
    Tuple,
    Union,
)

rxn_var_separator = "___var"




def run_reactants(
    reaction: AllChem.ChemicalReaction,
    reactants: Sequence[Chem.Mol],
) -> List[Chem.Mol]:
    """
    Return sanitized and canonicalized products from running reaction `rxn` with reactants `reacts`
    """
    reactants = tuple(reactants)

    rxn_products = []
    unique_smiles = set()

    product_sets: Tuple[Tuple[Mol, ...], ...] = reaction.RunReactants(reactants)
    for product_set in product_sets:
        for prods in product_set:
            # these prods can in principle
            # a) be equivalent if the reactants are symmetric h
            # b) contain multiple fragments.

            # we want to avoid both of these cases -- so we will only add unique smiles
            # and fail on multiple fragments

            fragments = rdmolops.GetMolFrags(prods)

            if len(fragments) > 1:
                raise ValueError(
                    "Multiple reaction products returned! This should not happen in DEL SIMULATOR -- check your reaction templates"
                )

            old_len = len(unique_smiles)
            smiles = Chem.MolToSmiles(
                prods
            )  # cannot get away from this single conversion

            unique_smiles.add(smiles)

            if len(unique_smiles) > old_len:
                Chem.SanitizeMol(prods)
                rxn_products.append(prods)

    return rxn_products




class ReactionGraph:
    def __init__(self, reaction_scheme: ReactionScheme) -> None:
        # Initialize the Graph topology and attributes from the reaction scheme
        self.graph = nx.node_link_graph(reaction_scheme.model_dump())

        if not nx.is_directed_acyclic_graph(self.graph):
            raise ValueError(
                f"The reaction scheme must be represented by a graph that is a directed acyclic graph. Edges are: { self.graph.edges()}"
            )

        self.root_node_id = next(nx.topological_sort(self.graph))

        self.leaf_nodes = [
            node for node in self.graph.nodes() if self.graph.out_degree(node) == 0
        ]

        logging.info("Initialized Reaction Graph")
        logging.info("Reaction templates must correspond to the order input nodes!")
        # fixme: optionally allow for symmetrized reactions, at a cost of performance

    @staticmethod
    @lru_cache(maxsize=1000)
    def _run_reaction(
        current_mol: Chem.Mol,
        added_bb: Chem.Mol | None,
        reaction: rdChemReactions.ChemicalReaction,
        return_all_products: bool = True,
    ) -> T.List[Chem.Mol]:
        """
        This function runs reaction encoded by reaction_smarts and returns the list of products

        if this is a one component reaction(i.e. a deprotection), run the reaction on the starting material
        if this is a two component reaction, append the incoming bb to the reactant list and run the reaction
        if this is a two-component reaction and the incoming bb is None, treat this as unreacted (e.g. a
        truncated/null-block synthon) and return only the starting material

        In all cases, prepend the unreacted starting material to the list of products
        """

        reactants = [current_mol]
        # FIXME assert that can only have 1 or 2 reactants in the reaction template
        if reaction.GetNumReactantTemplates() == 2 and added_bb is None:
            return [current_mol]

        if reaction.GetNumReactantTemplates() == 2:
            reactants.append(added_bb)

        products = run_reactants(
            reaction=reaction,
            reactants=reactants,
        )

        if None in products:
            logging.error(
                "Template: %s \n Reactants: %s \n",
                rdChemReactions.ReactionToSmarts(reaction),
                [Chem.MolToSmiles(r) for r in reactants],
            )
            raise ValueError(
                "Reaction failed: A malformed product was created! Check your reaction template!"
            )

        # The execution of the reaction may fail if the reaction of the current cycle
        # is not applicable to the reactants - e.g. if the starting reactant is a truncate (null-block) or
        # unreacted starting material from the previous step.

        # The reaction may also fail because it has to be defined forward -- the reaction template is not symmetrized

        #
        #      logging.debug("Reaction failed: %s  products", len(products))
        #  else:
        #      logging.debug("Reaction successful:  %s unique products", len(products))

        if not return_all_products:
            if len(products) > 1:
                return [current_mol, products[0]]

        return [current_mol, *products]

    def _collect_products_and_abundances(
        self,
        nsynthon_id: str,
        node_id: STEP_NAME,
        edge_outcome_counts: T.Dict[T.Tuple[STEP_NAME, STEP_NAME], T.Dict[str, int]],
    ) -> CollectedSynthonOutput:
        """
        Collects the products and their relative abundances
        from the terminal node of a traversal of the reaction path

        Args:
            nsynthon_id (str): The string id of the nsynthon.
            node_id (STEP_NAME): The terminal node of the traversal of the reaction path
            edge_outcome_counts: per-edge {"reacted"/"zero_product"/"skipped_no_bb": count}
                tallies accumulated while executing this nsynthon's reaction path

        Returns:
            CollectedSynthonOutput: An object containing the nsynthon ID and the collected products with their yields.
        """

        total_abundance = 0.0
        data = []

        node = self.graph.nodes[node_id]

        total_abundance = total_abundance + sum(node["data"].relative_product_fractions)
        if not math.isclose(1.0, total_abundance, rel_tol=1e-5):
            raise ValueError(
                f"Yields for nsynthon_id {nsynthon_id} do not add up to 1, {total_abundance}"
            )

        data.extend(
            (
                Chem.MolToSmiles(Chem.RemoveHs(p)),
                fraction,
            )
            for p, fraction in zip(
                node["data"].products, node["data"].relative_product_fractions
            )
        )

        return CollectedSynthonOutput(
            nsynthon_id=nsynthon_id,
            smiles_and_abundances=data,
            edge_outcome_counts=edge_outcome_counts,
        )

    def execute_reaction_graph(
        self, nsynthon_components: NSynthonComponents, return_all_products: bool = True
    ) -> CollectedSynthonOutput:
        """
        Executes the reaction graph for a given nsython


        Args:
            nsynthon_components (NSynthonComponents): The building blocks and id of the nsynthong

        Returns:
            CollectedSynthonOutput: The products and relative abundnances
        """
        node_data_dict = {}
        edge_outcome_counts: T.Dict[T.Tuple[STEP_NAME, STEP_NAME], T.Dict[str, int]] = (
            {}
        )

        # Populate the node attributes with the building blocks for the selected nsynthon
        path = list(nsynthon_components.building_blocks.keys())

        RDLogger.DisableLog("rdApp.*")
        try:
            for node_id in path:
                bb_to_add = nsynthon_components.building_blocks[node_id]

                products = [bb_to_add] if node_id == self.root_node_id else []
                relative_frac = [1.0] if node_id == self.root_node_id else []

                new_node_data = ReactionSchemeNodeData(
                    products=products,
                    bb_to_add=bb_to_add,
                    relative_product_fractions=relative_frac,
                )

                node_data_dict.update({node_id: new_node_data})

            nx.set_node_attributes(self.graph, node_data_dict, name="data")
        finally:
            RDLogger.EnableLog("rdApp.*")

        for edge in pairwise(path):
            source_node, target_node = (
                self.graph.nodes[edge[0]],
                self.graph.nodes[edge[1]],
            )

            reaction = self.graph.edges[edge]["reaction"]
            yield_gen = self.graph.edges[edge]["yield_generator"]

            products = []
            relative_product_fractions = []
            outcome_counts = Counter()

            for current_mol, relative_frac in zip(
                source_node["data"].products,
                source_node["data"].relative_product_fractions,
            ):
                _rxn_products = self._run_reaction(
                    current_mol,
                    target_node["data"].bb_to_add,
                    reaction,
                    return_all_products,
                )

                # Categorize this attempt so failure rates can be tracked per-edge across a
                # full generation run. A reaction with no building block supplied is a
                # deliberate skip (e.g. a truncated/null-block synthon in a multi-step scheme)
                # and is tracked separately from an attempted reaction that produced zero
                # products -- the latter is ambiguous per-call (it can mean the supplied
                # building block simply lacks the reactive handle, or that the reaction
                # template's reactant order doesn't match how it's invoked), but a persistently
                # high zero_product rate for one edge across many different building blocks is
                # a strong signal of the latter.
                if (
                    reaction.GetNumReactantTemplates() == 2
                    and target_node["data"].bb_to_add is None
                ):
                    outcome_counts["skipped_no_bb"] += 1
                elif len(_rxn_products) > 1:
                    outcome_counts["reacted"] += 1
                else:
                    outcome_counts["zero_product"] += 1

                products.extend(_rxn_products)

                relative_product_fractions.extend(
                    relative_frac
                    * yield_gen.generate_yields(
                        current_mol=current_mol,
                        bb_to_add=target_node["data"].bb_to_add,
                        products=_rxn_products,
                    )
                )

            edge_outcome_counts[edge] = dict(outcome_counts)

            _new_node_data = replace(
                target_node["data"],
                products=products,
                relative_product_fractions=relative_product_fractions,
            )
            target_node["data"] = _new_node_data

        return self._collect_products_and_abundances(
            nsynthon_components.nsynthon_id, path[-1], edge_outcome_counts
        )
