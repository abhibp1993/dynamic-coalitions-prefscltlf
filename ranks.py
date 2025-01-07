"""
Rank assignment to product game
"""

import pickle
from prefscltl2pdfa import PrefAutomaton, PrefScLTL
from pathlib import Path
from product import ProductState

# ======================================================================================================================
# MODIFY ONLY THIS BLOCK
# ======================================================================================================================
EXAMPLE = "example2"  # Folder name of your blocks world implementation
GAME_CONFIG_FILE = "blockworld_4b_3a.conf"


# ======================================================================================================================


def assign_ranks_to_aut(aut: PrefAutomaton):
    """
    Assigns rank to each preference automaton state.

    :param aut: Preference automaton.
    :return: (dict) Format: {arm-name: {aut-state: rank}}
    """
    pass


def assign_rank_to_state(ranks_aut: dict, state: ProductState):
    """
    Assigns rank to given product game state.

    :param ranks_aut: (dict) Format: {arm-name: {aut-state: rank}}
    :param state: (ProductState) A state of product game.
    :return: (dict) {product game state: rank}
    """
    pass


if __name__ == '__main__':
    # Load game config
    with open(Path(__file__).parent / EXAMPLE / "out" / GAME_CONFIG_FILE, "rb") as f:
        game_config = pickle.load(f)

    # Load product game
    with open(Path(__file__).parent / EXAMPLE / "out" / f"{game_config['name']}_product.pkl", "rb") as f:
        product_game = pickle.loads(f.read())

    # Load automata
    spec_dir = Path(__file__).parent / EXAMPLE / "specs"

    specs = dict()
    paut = dict()
    for i in range(len(game_config["arms"])):
        arm = game_config["arms"][i]
        spec = PrefScLTL.from_file(spec_dir / f"{game_config['specs'][arm]}.spec")
        aut = spec.translate()
        specs[arm] = spec
        paut[arm] = aut

    # Assign ranks to all automata
    ranks_aut = dict()
    for arm, aut in paut.items():
        ranks_aut[arm] = assign_ranks_to_aut(aut)

    # Assign ranks to product game states
    ranks = dict()
    for state in product_game.states():
        ranks[state] = assign_rank_to_state(ranks_aut, state)

    # Save game model as pickle file
    with open(Path(__file__).parent / EXAMPLE / "out" / f"{game_config['name']}_ranks.pkl", "wb") as f:
        f.write(pickle.dumps(ranks))
