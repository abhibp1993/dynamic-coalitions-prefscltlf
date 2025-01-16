from pathlib import Path

from prefscltl2pdfa import utils

# from model_generator import GameState
from product import ProductGame, PrefScLTL
from ranks import assign_ranks, assign_rank_to_state
from synthesis import synthesis

# ======================================================================================================================
# MODIFY ONLY THIS BLOCK
# ======================================================================================================================
EXAMPLE = "example4"  # Folder name of your blocks world implementation
from example4.game_model import BlocksWorld

GAME_CONFIG = {
    "name": f"blockworld_4b_3a_3l",  # Name is generated automatically. If you want custom name, uncomment this line.
    "blocks": ['b1', 'b2', 'b3', 'b4'],
    "arms": ['a1', 'a2', 'a3'],
    "partitions": {
        "a1": {"b1"},
        "a2": {"b2"},
        "a3": {"b3", "b4"},
    },
    "priority": ['a1', 'a2', 'a3'],
    "specs": {  # File type `.prefscltl` will be added automatically.
        "a1": "spec1",
        "a2": "spec2",
        "a3": "spec3",
    },
    "location": 3
}

CONSTRUCTION_CONFIG = {
    "out": Path(__file__).parent / EXAMPLE / "out",
    "show_progress": True,
    "debug": False,
    "check_state_validity": True
}

if not CONSTRUCTION_CONFIG["out"].exists():
    CONSTRUCTION_CONFIG["out"].mkdir(parents=False)

name = f"blockworld_{len(GAME_CONFIG['blocks'])}b_{len(GAME_CONFIG['arms'])}a"

if (CONSTRUCTION_CONFIG["out"] / f"{name}.conf").exists():
    raise FileExistsError(f"Game model with name '{name}' already exists. Please change the name in GAME_CONFIG.")

GAME_CONFIG["name"] = name
# Define specs
spec_dir = Path(__file__).parent / EXAMPLE / "specs"


# ======================================================================================================================


def main():
    game = BlocksWorld(
        name=GAME_CONFIG["name"],
        blocks=GAME_CONFIG["blocks"],
        arms=GAME_CONFIG["arms"],
        partitions=GAME_CONFIG["partitions"],
        priority=GAME_CONFIG["priority"],
        location=GAME_CONFIG["location"]
        # check_state_validity=CONSTRUCTION_CONFIG["check_state_validity"]
    )

    game_graph = game.build(
        build_labels=True,
        show_progress=CONSTRUCTION_CONFIG["show_progress"],
        debug=CONSTRUCTION_CONFIG["debug"]
    )

    # Load specs
    specs = dict()
    automata = dict()
    for i in range(len(GAME_CONFIG["arms"])):
        arm = GAME_CONFIG["arms"][i]
        spec = PrefScLTL.from_file(spec_dir / f"{GAME_CONFIG['specs'][arm]}.spec")
        aut = spec.translate()
        specs[arm] = spec
        automata[arm] = aut

    # Create product game
    product_game = ProductGame(
        name="ProductGame",
        game=game,
        automata=[automata[arm] for arm in GAME_CONFIG["arms"]]
    )

    product_game_graph = product_game.build(
        build_labels=True,
        show_progress=CONSTRUCTION_CONFIG["show_progress"],
        debug=CONSTRUCTION_CONFIG["debug"]
    )

    # Extract DFA list
    phi: dict = automata[GAME_CONFIG["arms"][0]].phi
    idx = [k for k in sorted(list(phi.keys())) if k != -1]
    dfa = [utils.scltl2dfa(phi[k]) for k in idx]

    # Assign ranks to all automata
    ranks_aut = dict()
    for arm, paut in automata.items():
        ranks_aut[arm] = assign_ranks(dfa, paut)

    # Assign ranks to product game states
    ranks = dict()
    for state in product_game.states(as_dict=True).keys():
        ranks[state] = assign_rank_to_state(ranks_aut, product_game.states(as_dict=True)[state])  # {0: (1, 2, 0)}

    # Run synthesis
    s = synthesis(product_game, conc_game, ranks_conc_game, costs, 3)

    # # Save game model as pickle file
    # with open(CONSTRUCTION_CONFIG["out"] / f"{GAME_CONFIG['name']}_product.pkl", "wb") as f:
    #     f.write(pickle.dumps(out))
    #
    # # Save game model as pickle file
    # out_dir = CONSTRUCTION_CONFIG["out"]
    #
    # with open(out_dir / f"{GAME_CONFIG['name']}.conf", "wb") as f:
    #     f.write(pickle.dumps(GAME_CONFIG))
    #
    # with open(out_dir / f"{GAME_CONFIG['name']}.pkl", "wb") as f:
    #     f.write(pickle.dumps(out))
