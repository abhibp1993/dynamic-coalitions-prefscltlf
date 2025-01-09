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


CONSTRUCTION_CONFIG = {
    "out": Path(__file__).parent / EXAMPLE / "out",
    "show_progress": True,
    "debug": False,
    "check_state_validity": True
}

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


def _assign_costs(conc_game, ranks, player, num_players):
    # Identify maximum rank
    max_rank = max(ranks.values())  # TODO: Check if this is correct

    # Iterate to find smallest rank that can be enforced by player from every state
    cost = {state: float("inf") for state in conc_game.states()}
    for rank in range(max_rank):
        # Compute final states at this rank
        final_states = None  # TODO: Implement

        # Compute sure winning states at this rank
        solver = SWinReach(  # TODO: Check
            game=concurrent_game,
            final=final_states,
            num_players=num_players,
            player=player,
        )
        solver.solve()
        winning_states = solver.winning_nodes[player]  # TODO: Check

        if rank == 0:
            pass  # TODO: Implement
        else:
            pass  # TODO: Implement

    return cost


def assign_costs(product_game, ranks, num_players):
    """ Assign a vector-valued cost to each state in concurrent game version of input product game. """
    # Construct concurrent game
    conc_game = construct_conc_game(product_game)

    # Assign costs
    cost = dict()
    for player in range(num_players):
        cost[player] = _assign_costs(conc_game, ranks, player, num_players)

    cost_vector = dict()
    for state in conc_game.states():
        cost_vector[state] = tuple(cost[player][state] for player in range(num_players))

    return cost_vector


if __name__ == '__main__':
    # Load game config
    with open(Path(__file__).parent / EXAMPLE / "out" / GAME_CONFIG_FILE, "rb") as f:
        game_config = pickle.load(f)

    # Load product game
    with open(Path(__file__).parent / EXAMPLE / "out" / f"{game_config['name']}_product.pkl", "rb") as f:
        product_game = pickle.loads(f.read())

    # Load ranks
    with open(Path(__file__).parent / EXAMPLE / "out" / f"{game_config['name']}_ranks.pkl", "rb") as f:
        ranks = pickle.loads(f.read())

    # Compute costs for each player
    costs = assign_costs(product_game, ranks, 3)

    # Save costs
    with open(Path(__file__).parent / EXAMPLE / "out" / f"{game_config['name']}_costs.pkl", "wb") as f:
        pickle.dump(costs, f)

    # # Construct a concurrent game
    # concurrent_game = construct_conc_game(product_game)
    #
    # solver = SWinReach(
    #     game=concurrent_game,
    #     final=[165, 237, 103, 334, 65],
    #     num_players=3,
    #     player=1,
    # )
    # solver.solve()
    # print(solver.winning_nodes[1])
