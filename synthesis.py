import pickle
import random

from ggsolver.solvers.dtptb import SWinReach
from pathlib import Path
from product import ProductState
from pprint import pprint
from ggsolver.solvers.cdg import *
from ggsolver.game import GraphGame


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

#This function also takes ranks, actually it will take ranks for the concurrent game states
def _strategy_given_rank(rank, product_game, conc_game, values, n_players,ranks):
    # Get P1 states with given ranks
    states = set()  #
    states={id for id, value in ranks.items() if value[0] <= rank}

    # Fix point computation
    d=pre(states, conc_game)
    set_u = pre(states, conc_game).keys()-states

    # TODO (Use concurrent game)
    costs = dict()
    while set_u:
        # Iterate over all states in set_u
        for u in set_u:
            for players, act in get_coalition_actions(conc_game, u):
                if len(players) > 1:
                    # Decouple players
                    _, player_i = players

                    # Get next states given coalition action
                    next_states_under_a = partial_transition(conc_game, u, players, act, values)

                    # If coalition is NOT rational for player i, eliminate coalition action
                    if values[u][player_i] < max(values[v][player_i] for v in next_states_under_a):
                        pass        # TODO

                    else:
                        pass        # TODO. Update max player-i cost that can be guaranteed

                # Compute costs for all non-coalitional player
                for player_j in set(range(n_players)) - players:
                    pass        # TODO. Update cost for player j

                # Update costs dictionary: {state: {coalition-action: {non-coalitional-action: cost}}
                # TODO

        # Eliminate states with no enabled actions (use costs dictionary)
        # For surviving states, update max costs for all players.

        # Update Vk
        # Break condition

        # Update set_u
        set_u = None  # Pre(Vk) - Vk
        #states= states | set_u








    return None, None


def get_coalition_actions(game, u):
    en_actions = defaultdict(set)
    en_actions = set()
    # For each state in the game graph

    # Iterate over all out-edges of the state
    for _, _, act, _ in game.transitions(from_state=u):
        players=len(act)
            # Update enabled actions dictionary
        en_actions.add((1,act[0]))
        for i in range(players-1):
            en_actions.add(((1,i+2), (act[0],act[i+1])))
    return en_actions


def partial_transition(conc_game, u, players, act, values):
    return set()


def pre(set_u,conc_game):
    frontier = dict()

    for state in conc_game.predecessors(set_u):
        eliminated_actions = set()
        for (players,act_i) in get_coalition_actions(conc_game, state):
            for _, next_state, action, _ in conc_game.transitions(from_state=state):
                if players ==1:
                    if act_i == action[0] and next_state not in set_u:
                        eliminated_actions.add((1,act_i))

                else:
                    _, player_i = players
                    if act_i[0] == action[0] and act_i[1] == action[player_i-1] and next_state not in set_u:
                        eliminated_actions.add((players, act_i))
        if get_coalition_actions(conc_game, state)  != eliminated_actions:
            frontier[state]=get_coalition_actions(conc_game, state)-eliminated_actions

    return frontier


# def pre(set_u,conc_game):
#     frontier = set(conc_game.predecessors(set_u))
#
#     for state in conc_game.predecessors(set_u):
#         eliminated_actions = set()
#         for (players,act_i) in get_coalition_actions(conc_game, state):
#             for _, next_state, action, _ in conc_game.transitions(from_state=state):
#                 if players ==1:
#                     if act_i == action[0] and next_state not in set_u:
#                         eliminated_actions.add((1,act_i))
#
#                 else:
#                     _, player_i = players
#                     if act_i[0] == action[0] and act_i[1] == action[player_i-1] and next_state not in set_u:
#                         eliminated_actions.add((players, act_i))
#         if get_coalition_actions(conc_game, state)  == eliminated_actions:
#             frontier.discard(state)
#
#     return frontier




def synthesis(product_game, conc_game, ranks, values, n_players):
    # Compute max rank
    max_rank =max(value[0] for value in ranks.values())  # TODO

    # Iterate over all rank until initial state is winning
    for rank in range(max_rank):
        win_states, strategy = _strategy_given_rank(rank, product_game, conc_game, values, n_players,ranks)

    return set()

if __name__ == '__main__':
    # Load game config
    # Load product game
    # Load concurrent game
    # Load rank function
    # Load values
    # Run synthesis

    # Load game config
    with open(Path(__file__).parent / EXAMPLE / "out" / GAME_CONFIG_FILE, "rb") as f:
        game_config = pickle.load(f)

    # Load product game
    with open(Path(__file__).parent / EXAMPLE / "out" / f"{game_config['name']}_product.pkl", "rb") as f:
        product_game = pickle.loads(f.read())

    # Load ranks
    with open(Path(__file__).parent / EXAMPLE / "out" / f"{game_config['name']}_ranks.pkl", "rb") as f:
        ranks = pickle.loads(f.read())

    with open(Path(__file__).parent / EXAMPLE / "out" / f"{game_config['name']}_costs.pkl", "rb") as f:
        costs = pickle.loads(f.read())

    with open(Path(__file__).parent / EXAMPLE / "out" / f"{game_config['name']}_conc_game.pkl", "rb") as f:
        conc_game = pickle.loads(f.read())

    with open(Path(__file__).parent / EXAMPLE / "out" / f"{game_config['name']}_ranks_conc_game.pkl", "rb") as f:
        ranks_conc_game = pickle.loads(f.read())


    # with open(CONSTRUCTION_CONFIG["out"] / f"{game_config['name']}.pkl", "rb") as f:
    # game = pickle.loads(f.read())
    # # Load ranks
    # with open(Path(__file__).parent / EXAMPLE / "out" / f"{game_config['name']}_ranks.pkl", "rb") as f:
    #     ranks = pickle.loads(f.read())

    # Compute costs for each player

    s=synthesis(product_game, conc_game, ranks_conc_game, costs, 3)




