import json
from pathlib import Path
from unittest.mock import patch
import pytest
import networkx as nx
from rdkit import Chem
from rdkit.Chem import rdChemReactions

from del_simulator.reaction_graph import ReactionGraph
from del_simulator.core import (
    ReactionScheme,
    NSynthonComponents,
)


@pytest.fixture(scope="module")
def reaction_scheme():
    with open(
        Path(__file__).parent / "resources/dag0.json",
        "r",
    ) as jsonfile:
        reaction_scheme = ReactionScheme.model_validate(json.load(jsonfile))
    return reaction_scheme


def test_reaction_graph_init(reaction_scheme):
    g = ReactionGraph(reaction_scheme)

    assert g.graph.number_of_nodes() == 3
    assert g.graph.number_of_edges() == 2


def test_reactant_order_is_not_symmetrized(reaction_scheme):
    """
    Reaction templates are deliberately NOT symmetrized to reactant order: RDKit's
    RunReactants requires reactants in the exact order the SMARTS template lists them,
    and a template with reactants in the wrong order should silently produce no real
    products rather than reacting anyway.
    """
    g = ReactionGraph(reaction_scheme)

    source, target, data = next(e for e in g.graph.edges(g.root_node_id, data=True))
    first_reaction_smarts = data["reaction_template"]

    # reaction templates match against explicit hydrogens (e.g. "[NX3:...][#1]"), so
    # building blocks need Chem.AddHs -- matching how clean_building_blocks prepares them
    nsynthon_def = NSynthonComponents(
        nsynthon_id="0_0_0",
        building_blocks={
            "B0": Chem.AddHs(Chem.MolFromSmiles("CCCCCN")),
            "B1": Chem.AddHs(Chem.MolFromSmiles("O=C(O)Cc1cc(Br)ccc1[N+](=O)[O-]")),
            "B2": None,
        },
    )

    fwd = g.execute_reaction_graph(nsynthon_def)

    # the correct reactant order produces a real (amidated) product in addition to
    # the unreacted starting material
    assert len(fwd.smiles_and_abundances) == 2

    # flip the reactant ordering of the template, and rebuild the RDKit reaction object
    # accordingly -- the "reaction" edge attribute (not "reaction_template") is what
    # execute_reaction_graph actually runs
    r = first_reaction_smarts.split(">>")[0].split(".")
    flipped_smarts = f"{r[1]}.{r[0]}>>{first_reaction_smarts.split('>>')[1]}"
    nx.set_edge_attributes(
        g.graph,
        {
            (source, target): {
                "reaction_template": flipped_smarts,
                "reaction": rdChemReactions.ReactionFromSmarts(flipped_smarts),
            }
        },
    )

    bwd = g.execute_reaction_graph(nsynthon_def)

    # with reactants in the wrong order, RunReactants finds no match -- only the
    # unreacted starting material comes through
    assert len(bwd.smiles_and_abundances) == 1
    assert bwd.smiles_and_abundances[0][1] == 1.0


def test_execute_reaction_graph_does_not_rebuild_edge_attributes_per_product(
    reaction_scheme,
):
    """
    reaction/yield_generator edge attributes are constant for a whole edge, but used to be
    looked up via nx.get_edge_attributes(self.graph, ...)[edge] inside the loop over each
    incoming product from the previous step -- which rebuilds a dict over every edge in the
    graph on every call, even though a source node can already have multiple products by
    this point (the second edge below iterates twice, since B1 has 2 products after the
    amidation step). Direct edge-attribute access (self.graph.edges[edge][...]) computed
    once per edge replaces this.
    """
    g = ReactionGraph(reaction_scheme)

    nsynthon_def = NSynthonComponents(
        nsynthon_id="0_0_0",
        building_blocks={
            "B0": Chem.AddHs(Chem.MolFromSmiles("CCCCCN")),
            "B1": Chem.AddHs(Chem.MolFromSmiles("O=C(O)Cc1cc(Br)ccc1[N+](=O)[O-]")),
            "B2": None,
        },
    )

    with patch(
        "del_simulator.reaction_graph.nx.get_edge_attributes",
        wraps=nx.get_edge_attributes,
    ) as mocked:
        result = g.execute_reaction_graph(nsynthon_def)

    assert mocked.call_count == 0
    # sanity check: still produced real output, not just a no-op
    assert len(result.smiles_and_abundances) == 2
