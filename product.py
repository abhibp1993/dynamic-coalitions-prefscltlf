import itertools
import pickle
import spot

from ggsolver.generators import tsys
from ggsolver.generators.tsys.cls_state import *
from typing import Iterable, List
from prefscltl2pdfa import PrefAutomaton, PrefScLTL
from pprint import pprint
from pathlib import Path

# ======================================================================================================================
# MODIFY ONLY THIS BLOCK
# ======================================================================================================================
EXAMPLE = "example2"  # Folder name of your blocks world implementation
GAME_CONFIG_FILE = "blockworld_4b_3a.conf"
from example2.game_model import BlocksWorld

CONSTRUCTION_CONFIG = {
    "out": Path(__file__).parent / EXAMPLE / "out",
    "show_progress": True,
    "debug": False,
    "check_state_validity": True
}


# ======================================================================================================================

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
    def __init__(self, name, game, automata: List[PrefAutomaton], skip_sa_state_check=False):
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
        return ["TBD"]

    def states(self):
        return [
            ProductState(game_state=s, semi_aut_state=q)
            for s, q in itertools.product(self._game.states(), self._automata[0].get_states())
        ]

    def actions(self, state):
        s = state.game_state()
        # q = state.semi_aut_state()
        return {a for _, _, a, _ in self._game.transitions(from_state=s)}

    def delta(self, state, action):
        s = state.game_state()
        q = state.semi_aut_state()

        s_next = None
        for _, t, a, p in self._game.transitions(from_state=s):
            if a == action:
                s_next = t
                break

        label = self._game.get_label(state=s_next, is_id=True)
        subs_map = {p: True if p in label else False for p in self._game.atoms()}
        for cond, q_next in self._automata[0].transitions[q].items():
            if spot_eval(cond, subs_map):
                return ProductState(s_next, q_next)

    def atoms(self):
        return {str(i) for i in self._automata[0].get_states()}
        #return set()
    def label(self, state):
        return set(str(state.semi_aut_state()))
        #return set()


if __name__ == '__main__':
    # Load game config
    with open(CONSTRUCTION_CONFIG["out"] / GAME_CONFIG_FILE, "rb") as f:
        game_config = pickle.load(f)

    with open(CONSTRUCTION_CONFIG["out"] / f"{game_config['name']}.pkl", "rb") as f:
        game = pickle.loads(f.read())

    # Define specs
    spec_dir = Path(__file__).parent / EXAMPLE / "specs"

    specs = dict()
    paut = dict()
    for i in range(len(game_config["arms"])):
        arm = game_config["arms"][i]
        spec = PrefScLTL.from_file(spec_dir / f"{game_config['specs'][arm]}.spec")
        aut = spec.translate()
        specs[arm] = spec
        paut[arm] = aut

    # Create product game
    product_game = ProductGame(
        name="ProductGame",
        game=game,
        automata=[paut[arm] for arm in game_config["arms"]]
    )

    out = product_game.build(
        build_labels=True,
        show_progress=CONSTRUCTION_CONFIG["show_progress"],
        debug=CONSTRUCTION_CONFIG["debug"]
    )

    # Save game model as pickle file
    with open(CONSTRUCTION_CONFIG["out"] / f"{game_config['name']}_product.pkl", "wb") as f:
        f.write(pickle.dumps(out))
