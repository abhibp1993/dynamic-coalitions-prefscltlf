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
EXAMPLE = "example2"  # Folder nPrefAutomatoname of your blocks world implementation
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
    infinite_rank = set()

    # Create a set of all states in aut
    unassigned = aut.states.values()

    #   For each state (which is a tuple of form (q1, q2, ...)),
    #   identify if at least some qi is accepting in corresponding DFA.
    for state in unassigned:
        # Get state representation (see https://akulkarni.me/docs/prefltlf2pdfa/prefltlf2pdfa.html#prefltlf2pdfa.prefltlf.PrefAutomaton.get_states)
        # Check whether i-th component of state representation is accepting in i-th DFA.
        # If not, add state to infinite_rank set.
        cnt=0
        for i in range(len(aut.dfa)):
            if state[i] not in aut.dfa[i]['final_states']:
                cnt+=1

        if cnt == len(aut.dfa):
            infinite_rank.add(state)

    # Remove states with infinite rank from unassigned set
    unassigned -= infinite_rank

    # Assign ranks to unassigned states
    while unassigned:
        assigned = set.union(set(), *ranks)
        this_rank = set()

        for node in unassigned:
            neighbors = set(aut.pref_graph.edges[node]) - assigned - {node}
            if not neighbors:
                this_rank.add(node)

        ranks.append(this_rank)
        unassigned -= this_rank

    return ranks


def assign_rank_to_state(ranks_aut: dict, state: ProductState):
    """
    Assigns rank to given product game state.

    :param ranks_aut: (dict) Format: {arm-name: {aut-state: rank}}
    :param state: (ProductState) A state of product game.
    :return: (dict) {product game state: rank}
    """
    # TODO. Entire function
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
    for state in product_game.states():
        ranks[state] = assign_rank_to_state(ranks_aut, state)

    # Save game model as pickle file
    with open(Path(__file__).parent / EXAMPLE / "out" / f"{game_config['name']}_ranks.pkl", "wb") as f:
        f.write(pickle.dumps(ranks))
