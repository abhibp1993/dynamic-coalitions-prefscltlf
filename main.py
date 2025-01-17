import pickle
from pathlib import Path

from loguru import logger
from prefscltl2pdfa import PrefScLTL, utils

from algorithm1 import assign_costs, assign_ranks
# from model_generator import GameState
from product import ProductGame
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
    "specs": Path(__file__).parent / EXAMPLE / "specs",
    "show_progress": True,
    "debug": False,
    "check_state_validity": True,
    "skip_model_generation": True,
    "skip_product": True,
    "skip_ranks": True,
    "skip_costs": True,
    "skip_synthesis": False,
}

if not CONSTRUCTION_CONFIG["out"].exists():
    CONSTRUCTION_CONFIG["out"].mkdir(parents=False)

name = f"blockworld_{len(GAME_CONFIG['blocks'])}b_{len(GAME_CONFIG['arms'])}a"

# if (CONSTRUCTION_CONFIG["out"] / f"{name}.conf").exists():
#     raise FileExistsError(f"Game model with name '{name}' already exists. Please change the name in GAME_CONFIG.")

GAME_CONFIG["name"] = name
# Define specs
spec_dir = Path(__file__).parent / EXAMPLE / "specs"


# ======================================================================================================================

def load_config(skip_duplication_chk=True):
    """
    Checks for duplicate config files. If duplicate files found, raises error.

    :return:
    """
    out_dir = CONSTRUCTION_CONFIG["out"]
    if not skip_duplication_chk and (out_dir / f"{GAME_CONFIG['name']}.conf").exists():
        raise FileExistsError("Game config with name '{name}' already exists. Aborting.")

    with open(out_dir / f"{GAME_CONFIG['name']}.conf", "wb") as f:
        f.write(pickle.dumps(GAME_CONFIG))

    return GAME_CONFIG


def main_model_generator(game_config, construction_config):
    game = BlocksWorld(
        name=game_config["name"],
        blocks=game_config["blocks"],
        arms=game_config["arms"],
        partitions=game_config["partitions"],
        priority=game_config["priority"],
        location=game_config["location"]
        # check_state_validity=CONSTRUCTION_CONFIG["check_state_validity"]
    )
    game_graph = game.build(
        build_labels=True,
        show_progress=construction_config["show_progress"],
        debug=construction_config["debug"]
    )

    # Save game model as pickle file
    with open(construction_config["out"] / f"{game_config['name']}.pkl", "wb") as f:
        f.write(pickle.dumps(game_graph))

    # Return
    return game_graph


def main_product(game_graph, game_config, construction_config):
    # Define specs
    spec_dir = construction_config["specs"]

    specs = dict()
    automata = dict()
    for i in range(len(game_config["arms"])):
        arm = game_config["arms"][i]
        spec = PrefScLTL.from_file(spec_dir / f"{game_config['specs'][arm]}.spec")
        aut = spec.translate()
        specs[arm] = spec
        automata[arm] = aut

    # Create product game
    product_game = ProductGame(
        name=f"ProductGame({game_config['name']})",
        game=game_graph,
        automata=[automata[arm] for arm in game_config["arms"]]
    )

    product_game_graph = product_game.build(
        build_labels=True,
        show_progress=CONSTRUCTION_CONFIG["show_progress"],
        debug=CONSTRUCTION_CONFIG["debug"]
    )

    # Save game model as pickle file
    with open(CONSTRUCTION_CONFIG["out"] / f"{game_config['name']}_product.pkl", "wb") as f:
        f.write(pickle.dumps(product_game_graph))

    return product_game_graph


def main_ranks(product_game, game_config, construction_config):
    # Load automata
    spec_dir = construction_config["specs"]

    specs = dict()
    automata = dict()
    for i in range(len(game_config["arms"])):
        arm = game_config["arms"][i]
        spec = PrefScLTL.from_file(spec_dir / f"{game_config['specs'][arm]}.spec")
        aut = spec.translate()
        specs[arm] = spec
        automata[arm] = aut

    # Extract DFA list
    phi: dict = automata[game_config["arms"][0]].phi
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

    # Save game model as pickle file
    with open(Path(__file__).parent / EXAMPLE / "out" / f"{game_config['name']}_ranks.pkl", "wb") as f:
        f.write(pickle.dumps(ranks))

    return ranks


def main_costs(product_game, ranks, game_config, construction_config):
    # Compute costs for each player
    costs = assign_costs(product_game, ranks, 3)

    # Save costs
    with open(construction_config["out"] / f"{game_config['name']}_costs.pkl", "wb") as f:
        pickle.dump(costs, f)

    return costs


def main():
    # Load configuration
    game_cfg = load_config(skip_duplication_chk=True)

    # Run model generation
    model_fpath = CONSTRUCTION_CONFIG["out"] / f"{game_cfg['name']}.pkl"
    if CONSTRUCTION_CONFIG.get("skip_model_generation", False) and model_fpath.exists():
        with open(model_fpath, "rb") as f:
            game_graph = pickle.load(f)
        logger.info(f'Loaded game graph from {model_fpath}')
    else:
        game_graph = main_model_generator(game_cfg, CONSTRUCTION_CONFIG)

    # Run product construction
    product_fpath = CONSTRUCTION_CONFIG["out"] / f"{game_cfg['name']}_product.pkl"
    if CONSTRUCTION_CONFIG.get("skip_product", False) and product_fpath.exists():
        with open(product_fpath, "rb") as f:
            product_game_graph = pickle.load(f)
        logger.info(f'Loaded product game graph from {product_fpath}')
    else:
        product_game_graph = main_product(game_graph, game_cfg, CONSTRUCTION_CONFIG)

    # Run ranks assignment
    ranks_fpath = CONSTRUCTION_CONFIG["out"] / f"{game_cfg['name']}_ranks.pkl"
    if CONSTRUCTION_CONFIG.get("skip_ranks", False) and ranks_fpath.exists():
        with open(ranks_fpath, "rb") as f:
            ranks = pickle.load(f)
        logger.info(f'Loaded ranks from {ranks_fpath}')
    else:
        ranks = main_ranks(product_game_graph, game_cfg, CONSTRUCTION_CONFIG)

    # Run algorithm 1 to generate costs and concurrent game
    costs_fpath = CONSTRUCTION_CONFIG["out"] / f"{game_cfg['name']}_costs.pkl"
    if CONSTRUCTION_CONFIG.get("skip_costs", False) and costs_fpath.exists():
        with open(costs_fpath, "rb") as f:
            costs = pickle.load(f)
        logger.info(f'Loaded costs from {costs_fpath}')
    else:
        costs = assign_costs(product_game_graph, ranks, 3)

    # Run synthesis
    win_states, win_costs = synthesis(product_game_graph, ranks, costs, 3)

    win_states_fpath = CONSTRUCTION_CONFIG["out"] / f"{game_cfg['name']}_win_states.pkl"
    win_costs_fpath = CONSTRUCTION_CONFIG["out"] / f"{game_cfg['name']}_win_costs.pkl"
    with open(win_states_fpath, "wb") as f:
        pickle.dump(win_states, f)
    with open(win_costs_fpath, "wb") as f:
        w_costs = {k: {k1: {k2: v2 for k2, v2 in v1.items()} for k1, v1 in v.items()} for k, v in win_costs.items()}
        pickle.dump(w_costs, f)


if __name__ == '__main__':
    main()
