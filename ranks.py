"""
Rank assignment to product game
"""

import pickle
from prefscltl2pdfa import PrefAutomaton, PrefScLTL
from pathlib import Path
from product import ProductState
from typing import List
from prefscltl2pdfa import utils

# ======================================================================================================================
# MODIFY ONLY THIS BLOCK
# ======================================================================================================================
EXAMPLE = "example4"  # Folder nPrefAutomatoname of your blocks world implementation
GAME_CONFIG_FILE = "blockworld_4b_3a.conf"


# ======================================================================================================================


def assign_ranks(automata: List[dict], aut: PrefAutomaton):
    """
    Assigns rank to each preference automaton state.

    :param aut: Preference automaton.
    :return: (dict) Format: {arm-name: {aut-state: rank}}
    """
    # TODO. Entire function

    # Initialize list to store ranks
    ranks = list()

    # Create a set of all states in aut preference graph
    unassigned = aut.pref_graph.nodes()

    # Assign ranks to unassigned states
    while unassigned:
        assigned = set.union(set(), *ranks)
        this_rank = set()

        for node in unassigned:
            neighbors = set(aut.pref_graph.successors(node)) - assigned - {node}
            if not neighbors:
                this_rank.add(node)

        ranks.append(this_rank)
        unassigned -= this_rank

    # Assign ranks to semi-automaton states
    rank_state = []
    for rank_nodes in ranks:
        states_with_this_rank = set()
        for node in rank_nodes:
            states_with_this_rank.update(set(aut.pref_graph.nodes[node]['partition']))
        rank_state.append(states_with_this_rank)

    # Create dictionary of state to rank.
    ranks = dict()
    for i in range(len(rank_state)):
        for state in rank_state[i]:
            ranks[state] = i

    return ranks


def assign_rank_to_state(ranks_aut: dict, state: ProductState):
    """
    Assigns rank to given product game state.

    :param ranks_aut: (dict) Format: {arm-name: {aut-state: rank}}
    :param state: (ProductState) A state of product game.
    :return: (tuple) ranks for sorted arm names.
    """
    sorted_arms = sorted(ranks_aut.keys())
    return tuple(ranks_aut[arm][state.semi_aut_state()] for arm in sorted_arms)


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
