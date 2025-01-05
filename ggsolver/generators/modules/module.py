from collections.abc import Iterable
from ggsolver.generators.modules.cls_state import *
from ggsolver.generators.modules.builder import *
import ggsolver.game as game


class Module:
    def __init__(self, name, model_type, **kwargs):
        # Input parameters
        self._model_type = model_type
        self._name = name

        # Instance variables
        self._state_vars = dict()  # dict: {var-name: var-domain}
        self._state_validators = dict()  # dict: {validator-name: validator-func(state) -> boolean}
        self._init_states = set()
        self._actions = dict()  # dict: {act-name: act-func(state) -> next-state(s)}
        self._enabled_actions = None  # function/None: func(state) -> subset of self._actions.keys()
        self._atoms = dict()  # dict: {atom-name: atom-func(state) -> boolean}

    def __str__(self):
        return f"Module(type={self._model_type}, name={self._name})"

    @classmethod
    def define(cls, **kwargs):
        """
        Define a module with the given keyword parameters.
        :param kwargs:
            - state_vars: dict: {var-name: var-domain}
            - actions: dict: {act-name: act-func(state) -> next-state(s)}
            - enabled_actions: function/None: func(state) -> subset of actions.keys()
            - atoms: dict: {atom-name: atom-func(state) -> boolean}
            - init_states: list[dict]: [{var-name: var-value}]
            - state_validators: dict: {validator-name: validator-func(state) -> boolean}

        :return:
        """
        raise NotImplementedError

    @classmethod
    def define_explicit(cls, **kwargs):
        raise NotImplementedError

    def add_state_var(self, name: str, domain: Iterable):
        """
        Add a state variable with the given name and domain.
        :param name: (str) Name of variable.
        :param domain: (Iterable) Hashable iterable over domain variables.
        """
        assert name not in self._state_vars, f"State variable {name} already exists."
        self._state_vars[name] = set(domain)

    def add_state_vars(self, **kwargs):
        """
        Add multiple state variables with the given names and domains.

        :param kwargs: (dict) {var-name: var-domain} pairs.
        """
        for name, domain in kwargs.items():
            self.add_state_var(name, domain)

    def add_action(self, name, func):
        """
        Add an action with the given name and function.
        :param name: (str) Name of action.
        :param func: (function) Function that takes a state and returns a next state or a set of next states.
        :return:
        """
        assert name not in self._actions, f"Action {name} already exists."
        self._actions[name] = func

    def add_actions(self, **kwargs):
        """
        Add multiple actions with the given names and functions.

        :param kwargs: (dict) {act-name: act-func} pairs.
        """
        for name, func in kwargs.items():
            self.add_action(name, func)

    def set_enabled_actions(self, func):
        self._enabled_actions = func

    def add_init_state(self, state=None, **value):
        """
        Add an initial state to the module.
        :param state: (State) State object.
        :param value: (dict) {var-name: var-value} pairs.
        """
        if state is not None:
            self._init_states.add(state)
        else:
            self._init_states.add(State(**value))

    def add_init_states(self, *args):
        """
        Add multiple initial states to the module.

        :param args: (list[State]) List of State objects.
        """
        for state in args:
            if isinstance(state, State):
                self.add_init_state(state=state)
            elif isinstance(state, dict):
                self.add_init_state(**state)
            else:
                raise ValueError(f"Invalid state type: {type(state)}")

    def add_atom(self, name, func):
        """
        Add a label with the given name and function.
        :param name: (str) Name of label.
        :param func: (function) Function that takes a state and returns a boolean.
        """
        assert name not in self._atoms, f"Label {name} already exists."
        self._atoms[name] = func

    def add_atoms(self, **kwargs):
        """
        Add multiple labels with the given names and functions.

        :param kwargs: (dict) {label-name: label-func} pairs.
        """
        for name, func in kwargs.items():
            self.add_atom(name, func)

    def add_state_validator(self, name, func):
        """
        Add a state validator with the given name and function.
        :param name: (str) Name of validator.
        :param func: (function) Function that takes a state and returns a boolean.
        """
        assert name not in self._state_validators, f"State validator {name} already exists."
        self._state_validators[name] = func

    def add_state_validators(self, **kwargs):
        """
        Add multiple state validators with the given names and functions.

        :param kwargs: (dict) {validator-name: validator-func} pairs.
        """
        for name, func in kwargs.items():
            self.add_state_validator(name, func)

    def state_vars(self):
        return self._state_vars

    def actions(self):
        return self._actions

    def enabled_actions(self, state):
        if self._enabled_actions is not None:
            return self._enabled_actions(state)
        return self._actions

    def state_validators(self):
        return self._state_validators

    def init_states(self):
        return self._init_states

    def atoms(self):
        return self._atoms

    def model_type(self):
        return self._model_type

    def name(self):
        return self._name

    def build(self, **kwargs):
        builder = Builder(self, **kwargs)
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
