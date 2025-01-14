import itertools
import pickle
# import sys
# sys.path.insert(0, '/opt/homebrew/Cellar/spot/2.12.1/blib/python3.13/')

import spot

from game_model_3 import *
from ggsolver.generators import tsys
from ggsolver.generators.tsys.cls_state import *
from typing import Iterable, List
import sys
sys.path.append('/Users/andkaanyilmaz/Desktop/PrefScLTL/prefscltl2pdfa')  # Adjust the path
from prefscltl2pdfa.prefscltl import PrefAutomaton, PrefScLTL
from pprint import pprint


class ProductState(State):
    def __init__(self, game_state, semi_aut_state):
        self._game_state = game_state
        self._sa_state = semi_aut_state
        super().__init__(obj=(self._game_state, self._sa_state))

    def __hash__(self):
        return hash(self._obj)

    def __str__(self):
        return f"State(s={self._game_state}, q={self._sa_state})"

    def __repr__(self):
        return f"State(s={self._game_state}, q={self._sa_state})"

    def __eq__(self, other: 'ProductState') -> bool:
        if isinstance(other, ProductState):
            return (
                    self._game_state == other._game_state and
                    self._sa_state == other._sa_state
            )
        return False

    def get_object(self):
        return self._obj

    def game_state(self):
        return self._game_state

    def semi_aut_state(self):
        return self._sa_state


def spot_eval(cond, true_atoms):
    """
    Evaluates a propositional logic formula given the set of true atoms.

    :param true_atoms: (Iterable[str]) A propositional logic formula.
    :return: (bool) True if formula is true, otherwise False.
    """

    # Define a transform to apply to AST of spot.formula.
    def transform(node: spot.formula):
        if node.is_literal():
            if "!" not in node.to_str():
                if node.to_str() in true_atoms:
                    return spot.formula.tt()
                else:
                    return spot.formula.ff()

        return node.map(transform)

    # Apply the transform and return the result.
    # Since every literal is replaced by true or false,
    #   the transformed formula is guaranteed to be either true or false.
    return True if transform(spot.formula(cond)).is_tt() else False


class ProductGame(tsys.TransitionSystem):
    def __init__(self, name, game: tsys.TransitionSystem, automata: List[PrefAutomaton], skip_sa_state_check=False):
        # Base constructor
        super().__init__(
            name=f"ProductGame({game.name()})",
            model_type=game.model_type(),
            is_qualitative=game.is_qualitative()
        )

        # Type checks
        assert all(isinstance(aut, PrefAutomaton) for aut in automata), f"All automata must be PrefAutomata objects."
        if not skip_sa_state_check:
            for i in range(len(automata) - 1):
                assert automata[i].get_states() == automata[i + 1].get_states(), \
                    f"Automata at {i}-th and {i + 1}-th index must have the same states."

        # Save game and list of preference automaton
        self._game = game
        self._automata = automata

    def state_vars(self):
        return None

    def states(self):
        return itertools.product(self._game.states(), self._automata[0].get_states())

    def actions(self, state):
        s, q = state
        return {a for _, _, a, _ in self._game.transitions(from_state=s)}

    def delta(self, state, action):
        s, q = state
        s_next = None
        for _, t, a, p in self._game.transitions(from_state=s):
            if a == action:
                s_next = t
                break

        label = self._game.get_label(state=s_next, is_id=True)
        subs_map = {p: True if p in label else False for p in self._game.atoms()}
        for cond, q_next in self._automata[0].transitions[q].items():
            if spot_eval(cond, subs_map):
                return s_next, q_next

    def atoms(self):
        return set()

    def label(self, state):
        return set()


if __name__ == '__main__':
    # Load game
    with open("game_model.pickle", "rb") as f:
        game = pickle.loads(f.read())

    # Define specs
    spec0 = PrefScLTL.from_file("spec0.spec")
    aut0 = spec0.translate()

    spec1 = PrefScLTL.from_file("spec2.spec")
    aut1 = spec1.translate()

    # Create product game
    product_game = ProductGame(
        name="ProductGame",
        game=game,
        automata=[aut0, aut1]
    )

    # Print states
    pprint(product_game.states())

    # Print actions for each state
    for state in list(product_game.states()):
        print(state)
        pprint(product_game.actions(state))
        print("==== ")

    # Check delta function
    for state in product_game.states():
        for act in product_game.actions(state):
            print(f"delta({state}, {act}) = {product_game.delta(state, act)}")
