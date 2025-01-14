"""
Generates a game on graph from a transition style model definition.


# TODO:
    1. Implement get_state and set_state function for pickling.
    2. Think about build parallelization [Not priority]

"""

from ggsolver import game
from ggsolver.generators.tsys.builder import Builder


class TransitionSystem:
    def __init__(self, name: str, model_type: game.ModelTypes, *, is_qualitative=False, **kwargs):
        self._name = name
        self._model_type = model_type
        self._is_qualitative = is_qualitative

    def state_vars(self):
        # TODO. Provide default
        pass

    def states(self):
        raise NotImplementedError

    def actions(self, state):
        raise NotImplementedError

    def delta(self, state, action):
        raise NotImplementedError

    def init_states(self):
        raise NotImplementedError

    def atoms(self):
        raise NotImplementedError

    def label(self, state):
        raise NotImplementedError

    def reward(self, state, action=None):
        raise NotImplementedError

    def turn(self, state):
        raise NotImplementedError

    def build(self, **options):
        builder = Builder(self, **options)
        return builder.build()

    def name(self):
        return self._name

    def model_type(self):
        return self._model_type

    def is_deterministic(self):
        return game.is_deterministic(self.model_type())

    def is_probabilistic(self):
        return game.is_probabilistic(self.model_type())

    def is_qualitative(self):
        return self._is_qualitative
