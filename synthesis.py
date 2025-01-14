import pickle
import random

from ggsolver.solvers.dtptb import SWinReach
from pathlib import Path
from product import ProductState
from pprint import pprint
from ggsolver.solvers.cdg import *
from ggsolver.game import GraphGame
from collections import defaultdict

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

# This function also takes ranks, actually it will take ranks for the concurrent game states
def _strategy_given_rank(rank, product_game, conc_game, values, n_players, ranks):
    # Get P1 states with given ranks
    states = set()  #
    states = {state for state, value in ranks.items() if value[0] <= rank}

    # Fix point computation

    # TODO (Use concurrent game)
    # this keeps track of backpropagated costs
    costs = {state: [float("inf"), float("inf")] for state in conc_game.states()}
    # this is general dictionary of state, coalition, non_coalition and the respective actions
    general_costs = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for state in states:
        costs[state] = [values[state][1], values[state][2]]

    while True:
        d = pre(states, conc_game)
        set_u = d.keys() - states
        # Iterate over all states in set_u
        survived_states = set()
        for u in set_u:
            for players, act in d[u]:
                # this is the vector appended candidate_costs
                c = [float("inf"), float("inf")]
                if players != 1:
                    # Decouple players
                    _, player_i = players

                    # Get next states given coalition action
                    next_states_under_a = partial_transition(conc_game, u, players, act)

                    # If coalition is NOT rational for player i, eliminate coalition action
                    if values[u][player_i - 1] < max(costs[v][player_i - 2] for v in next_states_under_a):
                        d[u].remove((players, act))
                        # This 'continue' is for if the joint action is not rationalizable, the rest of the iteration does not have to be done. Am I correct with this?
                        continue
                        # TODO

                    else:
                        c[player_i - 2] = max(costs[v][player_i - 2] for v in next_states_under_a)
                        # TODO. Update max player-i cost that can be guaranteed

                # Compute costs for all non-coalitional players
                non_coalition = set()
                if players == 1:
                    players = {players}
                for player_j in set(player + 1 for player in range(n_players)) - {player for player in players}:
                    actions_j = set()
                    non_coalitional_cost = dict()
                    for _, _, a, _ in conc_game.transitions(from_state=u):
                        actions_j.add(a[player_j - 1])

                    for a_j in actions_j:
                        players_with_j = tuple(player for player in players)
                        players_with_j = players_with_j + (player_j,)
                        if isinstance(act, tuple) and len(act) == 2:
                            act_with_j = act + (a_j,)  # Unpack the second tuple
                        else:
                            act_with_j = (act,) + (a_j,)
                        next_states_j = partial_transition(conc_game, u, players, act_with_j)
                        non_coalitional_cost[a_j] = max(costs[state][player_j - 2] for state in next_states_j)

                    min_key = min(non_coalitional_cost, key=non_coalitional_cost.get)
                    c[player_j - 2] = non_coalitional_cost[min_key]
                    non_coalition.add(min_key)
                # Update costs dictionary: {state: {coalition-action: {non-coalitional-action: cost}}
                # It automatically obtains the survived states
                survived_states.add(u)
                general_costs[u][act][tuple(non_coalition)] = c

        # Eliminate states with no enabled actions (use costs dictionary)
        all_c_vectors_set = set()
        # For surviving states, update max costs for all players.
        for act, non_coalitions in general_costs[u].items():
            for non_coalition, c in non_coalitions.items():
                # Convert the c list to a tuple and add it to the set
                all_c_vectors_set.add(tuple(c))

        max_costs = tuple(max(t[i] for t in all_c_vectors_set) for i in range(len(c)))
        costs[u] = max_costs
        # Update Vk
        # Break condition

        # Update set_u
        # set_u = None  # Pre(Vk) - Vk
        if len(survived_states) == 0:
            break
        states = states | survived_states

    return states, None


def get_coalition_actions(game, u):
    en_actions = defaultdict(set)
    en_actions = set()
    # For each state in the game graph

    # Iterate over all out-edges of the state
    for _, _, act, _ in game.transitions(from_state=u):
        players = len(act)
        # Update enabled actions dictionary
        en_actions.add((1, act[0]))
        for i in range(players - 1):
            en_actions.add(((1, i + 2), (act[0], act[i + 1])))
    return en_actions


def partial_transition(conc_game, u, players, act):
    next_states = set()

    if len(players) == 2:
        player_1, player_i = players
        for _, next_state, a, _ in conc_game.transitions(from_state=u):
            if act[0] == a[0] and act[1] == a[player_i - 1]:
                next_states.add(next_state)
    elif len(players) == 3:
        player_1, player_i, player_j = players
        for _, next_state, a, _ in conc_game.transitions(from_state=u):
            if act[0] == a[0] and act[1] == a[player_i - 1] and act[2] == a[player_j - 1]:
                next_states.add(next_state)
    else:
        for _, next_state, a, _ in conc_game.transitions(from_state=u):
            if act[0] == a[0]:
                next_states.add(next_state)
    return next_states


def pre(set_u, conc_game):
    frontier = dict()

    for state in conc_game.predecessors(set_u):
        eliminated_actions = set()
        for (players, act_i) in get_coalition_actions(conc_game, state):
            for _, next_state, action, _ in conc_game.transitions(from_state=state):
                if players == 1:
                    if act_i == action[0] and next_state not in set_u:
                        eliminated_actions.add((1, act_i))

                else:
                    _, player_i = players
                    if act_i[0] == action[0] and act_i[1] == action[player_i - 1] and next_state not in set_u:
                        eliminated_actions.add((players, act_i))
        if get_coalition_actions(conc_game, state) != eliminated_actions:
            frontier[state] = get_coalition_actions(conc_game, state) - eliminated_actions

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
    max_rank = max(value[0] for value in ranks.values())  # TODO

    # Iterate over all rank until initial state is winning
    for rank in range(max_rank):
        win_states, strategy = _strategy_given_rank(rank, product_game, conc_game, values, n_players, ranks)
        if conc_game.init_states() in win_states:
            break

    return win_states


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

    s = synthesis(product_game, conc_game, ranks_conc_game, costs, 3)
