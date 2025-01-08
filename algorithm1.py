import pickle
import random

from ggsolver.solvers.dtptb import SWinReach
from pathlib import Path
from product import ProductState
from pprint import pprint
from ggsolver.solvers.cdg import *
from ggsolver.game import GraphGame

# ======================================================================================================================
# MODIFY ONLY THIS BLOCK
# ======================================================================================================================
EXAMPLE = "example2"  # Folder nPrefAutomatoname of your blocks world implementation
GAME_CONFIG_FILE = "blockworld_4b_3a.conf"


# ======================================================================================================================


def construct_conc_game(game):
    conc_game = GraphGame(name="concurrent_game", model_type="cdg")

    # Add states
    conc_game.add_states({state for sid, state in game.states(as_dict=True).items() if state.turn() == 1})

    # Add Transitions
    for sid, state in conc_game.states(as_dict=True).items():
        for _, intermediate_st, _, _ in game.transitions(from_state=game.state2id(state)):
            for _, next_st, a, _ in game.transitions(from_state=intermediate_st):
                next_state_id = conc_game.state2id(game.id2state(next_st))
                conc_game.add_transition((sid, next_state_id, a, None), as_names=False)

    return conc_game


if __name__ == '__main__':
    # Load game config
    with open(Path(__file__).parent / EXAMPLE / "out" / GAME_CONFIG_FILE, "rb") as f:
        game_config = pickle.load(f)

    # Load product game
    with open(Path(__file__).parent / EXAMPLE / "out" / f"{game_config['name']}_product.pkl", "rb") as f:
        product_game = pickle.loads(f.read())

    # # Load ranks
    # with open(Path(__file__).parent / EXAMPLE / "out" / f"{game_config['name']}_ranks.pkl", "rb") as f:
    #     ranks = pickle.loads(f.read())

    # Construct a concurrent game
    concurrent_game = construct_conc_game(product_game)

    solver = SWinReach(
        game=concurrent_game,
        final=[165, 237, 103, 334, 65],
        num_players=3,
        player=1,
    )
    solver.solve()
    print(solver.winning_nodes[1])
