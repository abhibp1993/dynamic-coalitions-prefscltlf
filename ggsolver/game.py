import json
import pickle
from pathlib import Path
from typing import Iterable, Union, Any

import networkx as nx

ModelTypes = [
    'mdp',  # Markov Decision Process
    'dtptb',  # Deterministic Two-Player Turn-Based
    'csg',  # Concurrent Stochastic Games
    'cdg',  # Concurrent Deterministic Games
    'smg',  # Stochastic Multiplayer Games
]


def is_deterministic(model_type):
    if model_type in ["dtptb", "cdg"]:
        return True
    return False


def is_probabilistic(model_type):
    if model_type in ["mdp", "csg", "smg"]:
        return True
    return False


class Game:
    """
    Abstract class to represent a game.

    This class serves as a template for various types of games and provides methods to handle game states, transitions,
    and properties such as determinism, stochasticity, concurrency, and turn-based dynamics.

    :param name: Name of the game.
    :type name: str
    :param model_type: The type of the game model (e.g., "mdp", "dtptb", "cdg", "csg", "smg").
    :type model_type: ModelTypes (Enum or str)
    :param qualitative: Optional parameter to specify if the game is qualitative or quantitative.
    :type qualitative: bool, optional
    :param kwargs: Additional keyword arguments.
    """

    def __init__(self, name: str, model_type: ModelTypes, qualitative=None, **kwargs):
        """
        Initializes the Game object.

        :param name: Name of the game.
        :type name: str
        :param model_type: The type of the game model (e.g., "mdp", "dtptb", "cdg", "csg", "smg").
        :type model_type: ModelTypes (Enum or str)
        :param qualitative: Indicates whether the game is qualitative (default: None).
        :type qualitative: bool, optional
        :param kwargs: Additional arguments for further customization.
        :type kwargs: dict
        """
        self._name = name
        self._model_type = model_type
        self._qualitative = qualitative

    def add_state(self, state):
        """
        Adds a single state to the game.

        :param state: The state to be added.
        :type state: Hashable object
        :raises NotImplementedError: Abstract method to be implemented in a subclass.
        """
        raise NotImplementedError("Abstract method.")

    def add_states(self, states):
        """
        Adds multiple states to the game.

        :param states: A list of states to be added.
        :type states: Iterable[Hashable]
        :raises NotImplementedError: Abstract method to be implemented in a subclass.
        """
        raise NotImplementedError("Abstract method.")

    def add_transition(self, transition):
        """
        Adds a single transition to the game.

        :param transition: A transition represented as a tuple or any suitable structure.
            - If game is deterministic or qualitative stochastic, the transition is a tuple (source, target, action).
            - If game is (quantitative) stochastic, the transition is a tuple (source, target, action, probability).
        :type transition: tuple or any
        :raises NotImplementedError: Abstract method to be implemented in a subclass.
        """
        raise NotImplementedError("Abstract method.")

    def add_transitions(self, transitions):
        """
        Adds multiple transitions to the game.

        :param transitions: A list of transitions. A transition represented as a tuple or any suitable structure.
            - If game is deterministic or qualitative stochastic, the transition is a tuple (source, target, action).
            - If game is (quantitative) stochastic, the transition is a tuple (source, target, action, probability).
        :type transitions: Iterable[tuple]
        :raises NotImplementedError: Abstract method to be implemented in a subclass.
        """
        raise NotImplementedError("Abstract method.")

    def set_label(self, state, labels):
        """
        Assigns labels to a specific state.

        :param state: The state in the game to label.
        :type state: Hashable
        :param labels: A set or list of labels to be assigned to the state.
        :type labels: list or set
        :raises NotImplementedError: Abstract method to be implemented in a subclass.
        """
        raise NotImplementedError("Abstract method.")

    def get_label(self, state):
        """
        Retrieves the labels for a specific state.

        :param state: The state for which to get labels.
        :type state: Hashable
        :return: Labels associated with the state.
        :rtype: list or set
        :raises NotImplementedError: Abstract method to be implemented in a subclass.
        """

        raise NotImplementedError("Abstract method.")

    def get_state_id(self, state):
        """
        Returns the ID of the given state.

        :param state: The state to get the ID for.
        :type state: Hashable
        :return: The ID of the state.
        :rtype: int
        :raises NotImplementedError: Abstract method to be implemented in a subclass.
        """
        raise NotImplementedError("Abstract method.")

    def get_state(self, state_id):
        """
        Returns the state corresponding to the given ID.

        :param state_id: The ID of the state.
        :type state_id: int
        :return: The state corresponding to the ID.
        :rtype: Hashable
        :raises NotImplementedError: Abstract method to be implemented in a subclass.
        """
        raise NotImplementedError("Abstract method.")

    def get_trans_to(self, dst):
        """
        Returns all transitions leading to the given destination state.

        :param dst: The destination state.
        :type dst: Hashable
        :return: A list of transitions leading to the destination state.
        :rtype: list
        :raises NotImplementedError: Abstract method to be implemented in a subclass.
        """
        raise NotImplementedError("Abstract method.")

    def get_trans_from(self, src):
        """
        Returns all transitions originating from the given source state.

        :param src: The source state.
        :type src: Hashable
        :return: A list of transitions originating from the source state.
        :rtype: list
        :raises NotImplementedError: Abstract method to be implemented in a subclass.
        """
        raise NotImplementedError("Abstract method.")

    def is_deterministic(self):
        """
        Checks if the game model is deterministic.

        :note: The function uses the model type to determine if the game is deterministic.
            It does not check the actual game graph for determinism.
            The model type "dtptb", "cdg" is deterministic, while "mdp" and "csg" are not.

        :return: True if the model is deterministic, False otherwise.
        :rtype: bool
        """
        if self._model_type in ["dtptb", "cdg"]:
            return True
        return False

    def is_stochastic(self):
        """
        Checks if the game model is stochastic.

        :note: The function uses the model type to determine if the game is stochastic.
            It does not check the actual game graph for determinism.
            The model type "dtptb", "cdg" is deterministic, while "mdp" and "csg" are not.

        :return: True if the model is stochastic, False otherwise.
        :rtype: bool
        """
        if self._model_type in ["mdp", "csg"]:
            return True
        return False

    def is_concurrent(self):
        """
        Checks if the game model is concurrent.

        :note: The function uses the model type to determine if the game is concurrent.
            The model types "csg", "cdg" and "smg" are concurrent, while "mdp" and "dtptb" are turn-based.


        :return: True if the model is concurrent, False otherwise.
        :rtype: bool
        """
        if self._model_type in ["csg", "cdg"]:
            return True
        return False

    def is_turn_based(self):
        """
        Checks if the game model is turn-based.

        :note: The function uses the model type to determine if the game is turn-based.
            The model types "csg", "cdg" and "smg" are concurrent, while "mdp" and "dtptb" are turn-based.

        :return: True if the model is turn-based, False otherwise.
        :rtype: bool
        """
        if self._model_type in ["dtptb"]:
            return True
        return False

    def is_qualitative(self):
        """
        Checks if the game is qualitative.

        :note: The function uses the user-defined `qualitative` parameter in constructor to determine if the game is qualitative.

        :return: True if the game is qualitative, False otherwise.
        :rtype: bool
        """
        return self._qualitative

    def model_type(self):
        return self._model_type

    def name(self):
        return self._name

    def states(self, as_dict=False, as_names=False) -> Union[list, dict]:
        """
        Returns the list of all states in the game.

        :return: List of all states.
        :rtype: list or dict
        :raises NotImplementedError: Abstract method to be implemented in a subclass.
        """
        raise NotImplementedError("Abstract method.")

    def transitions(self):
        """
        Returns the list of all transitions in the game.

        :return: List of all transitions.
        :rtype: list
        :raises NotImplementedError: Abstract method to be implemented in a subclass.
        """
        raise NotImplementedError("Abstract method.")

    def save_states(self, fpath: Path):
        if fpath.suffix == '.sta':
            with open(fpath, "w") as f:
                for state_id, state in self.states(as_dict=True).items():
                    f.write(f"{state_id}:\t{state}\n")

        elif fpath.suffix == '.json':
            with open(fpath, "w") as f:
                json.dump(self.states(as_dict=True), f)

        elif fpath.suffix == '.pkl':
            with open(fpath, "w") as f:
                pickle.dump(self.states(as_dict=True), f)

    def save_transitions(self, fpath: Path, state_names=False):
        states_dict = self.states(as_dict=True)

        if fpath.suffix == '.tra':
            with open(fpath, "w") as f:
                for u, v, a, p in self.transitions():
                    if not state_names:
                        f.write(f"{u}, {v}, {a}, {p} \n")
                    else:
                        # This is beautified
                        f.writelines([
                            f"{u=}: \t{states_dict[u]}\n",
                            f"a: \t{a}\n",
                            f"p: \t{p}\n",
                            f"{v=}: \t{states_dict[v]}\n",
                            f"# --------------\n",
                        ])

        elif fpath.suffix == '.json':
            with open(fpath, "w") as f:
                json.dump(self.transitions(), f)

        elif fpath.suffix == '.pkl':
            with open(fpath, "w") as f:
                pickle.dump(self.transitions(), f)

    def check_compliance(self):
        """
        Validates the game according to its model type.
        The procedure uses the game graph to determine if the game definition complies with the model type.

        :raises ValueError: If the model type is unknown.
        """
        if self._model_type == "mdp":
            self._check_mdp()
        elif self._model_type == "dtptb":
            self._check_dtptb()
        elif self._model_type == "cdg":
            self._check_cdg()
        elif self._model_type == "csg":
            self._check_csg()
        elif self._model_type == "smg":
            self._check_smg()
        else:
            raise ValueError("Unknown model type.")

    def _check_mdp(self):
        raise NotImplementedError("TBD")

    def _check_dtptb(self):
        raise NotImplementedError("TBD")

    def _check_csg(self):
        raise NotImplementedError("TBD")

    def _check_smg(self):
        raise NotImplementedError("TBD")

    def _check_cdg(self):
        return True


class GraphGame(Game):
    """
    Class to represent a game on graph.

    This class extends the `Game` class and represents a game using a directed multigraph. States are represented as
    nodes, and transitions are represented as edges in the graph.

    :param name: Name of the game.
    :type name: str
    :param model_type: The type of the game model (e.g., "mdp", "dtptb", "cdg", "csg", "smg").
    :type model_type: ModelTypes (Enum or str)
    :param kwargs: Additional keyword arguments.
    :type kwargs: dict
    """

    def __init__(self, name: str, model_type: ModelTypes, **kwargs):
        """
        Initializes the `GraphGame` object, creates an empty directed multigraph, and sets up internal mappings.

        :param name: Name of the game.
        :type name: str
        :param model_type: The type of the game model (e.g., "mdp", "dtptb", "cdg", "csg", "smg").
        :type model_type: ModelTypes (Enum or str)
        :param kwargs: Additional arguments for further customization. Currently no keyword arguments are supported.
        :type kwargs: dict
        """

        super().__init__(name, model_type, **kwargs)
        self._graph = nx.MultiDiGraph()
        self._actions = set()
        self._atoms = set()
        self._inv_node_map = dict()
        self._node_id = 0

    @property
    def graph(self):
        """
        Returns a non-editable view of the game's graph.

        :return: A view of the graph.
        :rtype: nx.GraphView
        """
        return nx.graphviews.generic_graph_view(self._graph)

    def add_state(self, state):
        if self._inv_node_map.get(state, None) is None:
            # Add the state to the graph
            self._graph.add_node(self._node_id, state=state, label=set())

            # Update the inverse node map
            self._inv_node_map[state] = self._node_id

            # Increment the node id
            self._node_id += 1

            # Return the node id
            return self._node_id - 1

        # Else, return the node id
        return self._inv_node_map[state]

    def add_states(self, states):
        return [self.add_state(state) for state in states]

    def add_transition2(self, transition):
        """ Representation: (u, v, a, p) """
        # Unpack the transition
        if self.is_deterministic():
            if len(transition) == 2:
                src, dst = transition
                action = (src, dst)
                prob = None
            else:
                src, dst, action = transition
                prob = None
        else:
            if self.is_qualitative():
                src, dst, action = transition
                prob = None
            else:
                src, dst, action, prob = transition

        # Get the node ids
        src_id = self._inv_node_map.get(src)
        dst_id = self._inv_node_map.get(dst)

        # Add the transition to the graph
        self._graph.add_edge(src_id, dst_id, key=action, prob=prob)

        # Update action set
        self._actions.add(action)

    def add_transition(self, transition, as_names=True):
        """ Representation: (u, v, a, p) """
        # # Unpack the transition
        # if self.is_deterministic():
        #     if len(transition) == 2:
        #         src, dst = transition
        #         action = (src, dst)
        #         prob = None
        #     else:
        #         src, dst, action = transition
        #         prob = None
        # else:
        #     if self.is_qualitative():
        #         src, dst, action = transition
        #         prob = None
        #     else:
        #         src, dst, action, prob = transition

        src, dst, action, prob = transition
        if self.is_deterministic() or self.is_qualitative():
            prob = None

        # Get the node ids
        if as_names is True:
            src_id = self._inv_node_map.get(src)
            dst_id = self._inv_node_map.get(dst)
        else:
            src_id, dst_id = src, dst

        # Add the transition to the graph
        self._graph.add_edge(src_id, dst_id, key=action, prob=prob)

        # Update action set
        self._actions.add(action)

    def add_transitions(self, transitions):
        for transition in transitions:
            self.add_transition(transition)

    def get_label(self, state, is_id=False):
        if not is_id:
            state = self._inv_node_map.get(state)
        return self._graph.nodes[state]['label']

    def set_label(self, state, labels: Union[str, Iterable[str]]):
        state_id = self._inv_node_map.get(state)

        if isinstance(labels, str):
            self._graph.nodes[state_id]['label'].add(labels)
            self._atoms.update({labels})
        else:
            self._graph.nodes[state_id]['label'].update(set(labels))
            self._atoms.update(labels)

    def set_state_vars(self, var_names):
        """
        Sets the state variables for the game.

        :param var_names: A tuple or list of variable names.
        :type var_names: Iterable
        """
        self._graph.graph["state_vars"] = tuple(var_names)

    def set_init_state(self, state):
        """
        Sets the initial state(s) for the game.

        :param state: The initial state.
        :type state: Any
        """
        self._graph.graph["init_states"] = self._graph.graph.get("init_states", {state})

    def states(self, as_names=False, as_dict=False):
        if as_dict and as_names:
            raise ValueError("Cannot set both as_dict and as_names to true.")

        if as_names:
            return self._inv_node_map.keys()

        if as_dict:
            return {v: k for k, v in self._inv_node_map.items()}

        return self._graph.nodes()

    def state2id(self, state):
        return self._inv_node_map[state]

    def id2state(self, state_id):
        return self._graph.nodes[state_id]["state"]

    def actions(self):
        return self._actions

    def transitions(self, from_state=None, to_state=None, as_names=False):
        if as_names:
            raise NotImplementedError("as_names not implemented yet")

        if from_state and to_state:
            edges = self._graph.edges([from_state, to_state], data=True, keys=True)
            edges = [edge for edge in edges if edge[0] == from_state and edge[1] == to_state]
        elif from_state is not None:
            edges = self._graph.out_edges(from_state, data=True, keys=True)
        elif to_state is not None:
            edges = self._graph.in_edges(to_state, data=True, keys=True)
        else:
            edges = self._graph.edges(data=True, keys=True)

        return set((edge[0], edge[1], edge[2], edge[3]['prob']) for edge in edges)

    def state_vars(self):
        """
        Returns the state variables for the game.

        :return: A tuple of state variable names, or None if not set.
        :rtype: tuple or None
        """
        return self._graph.graph.get("state_vars", None)

    def num_states(self):
        return self._node_id

    def num_transitions(self):
        return self._graph.number_of_edges()

    def init_states(self):
        return self._graph.graph.get("init_states", set())

    def atoms(self):
        return self._atoms

    def predecessors(self, set_of_states: Iterable[Any]):
        return set.union(*list(map(set, (self._graph.predecessors(state) for state in set_of_states))))


class MatrixGame(Game):
    pass


class DictGame(Game):
    pass
