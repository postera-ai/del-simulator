from del_simulator.library_generator import LibraryGenerator
from del_simulator.core import ReactionScheme, BuildingBlocksSmiles
import logging
import json

from del_simulator.core import (
    LibraryGenerationConfig,
)
import logging


from del_simulator.utils.utils import load_config, parse_arguments

if __name__ == "__main__":

    args = parse_arguments()

    config = load_config(
        config_file_path=args.config_file_path, cli_config_dotlist=args.config_attrs
    )
    logging.basicConfig(level=config.loglevel.upper())

    library_gen_config = load_config(
        config_file_path=args.config_file_path,
        cli_config_dotlist=args.config_attrs,
        config_schema=LibraryGenerationConfig,
        config_section="library_generation",
    )

    logging.info("Constructing Library")

    with open(library_gen_config.building_block_path, "r") as jsonfile:
        building_blocks = BuildingBlocksSmiles.model_validate(json.load(jsonfile))

    with open(library_gen_config.reaction_scheme_path, "r") as jsonfile:
        reaction_scheme = ReactionScheme.model_validate(json.load(jsonfile))
    if not library_gen_config.return_all_products:
        logging.warning(
            "return_all_products is set to False, this will only return the first product for each reaction, omitting any additional products from nonselective templates/reactants."
        )

    library_generator = LibraryGenerator(
        raw_bbs=building_blocks,
        reaction_scheme=reaction_scheme,
        num_workers=library_gen_config.num_cpu,
        chunksize=library_gen_config.chunksize,
        building_block_subsets=library_gen_config.building_block_subsets,
        output_path=library_gen_config.output_path,
        return_all_products=library_gen_config.return_all_products,
    )
    output = library_generator.generate()
