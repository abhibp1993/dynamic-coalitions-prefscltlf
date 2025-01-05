"""
Generates a game on graph from a PRISM model.

(14 Sept 2024) The code is elementary. It generates a game graph from a PRISM model.
    No checks/tests have been run to check for bugs/corner cases.

Remark. It is recommended to avoid using filters since it is time-consuming.
    Instead, encode filters directly in prism model.


TODO: Develop an gridworld example.

TODO: Think whether we want to keep node/edge attributes separate from nx graph representation?
    It might make life easy for processing the graph.
    Idea: G stores only the bare-bones structure (state-ids, transitions, turn, label, reward).
        State and transition attributes used for human-readability are stored separately.
"""

import ast
import pathlib
import shutil
import subprocess
import networkx as nx

from ggsolver.game import GraphGame, ModelTypes
from loguru import logger


class PrismGame:
    """
    Class to represent a game using PRISM model.

    This class handles the parsing and processing of models defined in PRISM, which includes Markov Decision Processes (MDP),
    Stochastic Multiplayer Games (SMG), and other model types supported by PRISM. The class also manages the temporary
    workspace for storing intermediate files, as well as parsing and building a `GraphGame` representation of the PRISM model.

    Attributes:
        ModelTypes (list): Supported PRISM model types.
        _name (str): The name of the game.
        _model_type (str): The type of model (one of the types in `ModelTypes`).
        _prism_file (str): Path to the PRISM file.
        _prism_executable (str): Path to the PRISM executable for running the model.
        _qualitative (bool): Flag indicating whether the model is qualitative.
        _state_filter (callable): A filter function applied to the game states.
        _trans_filter (callable): A filter function applied to the game transitions.
        _tmp_folder (pathlib.Path): Path to the temporary folder used for intermediate files.
        _auto_cleanup (bool): Flag for automatic cleanup of temporary files upon object deletion.
        _force_clear (bool): Flag to forcefully clear the temporary folder if it already exists.
        _turn_var_name (str): Name of the turn variable, used in turn-based games.

    Methods:
        build(labels=False, state_rewards=False, trans_rewards=False, verbose=False, check_compliance=False):
            Builds the PRISM model and returns a `GraphGame` object representing the game graph.

        _run_prism(labels=False, state_rewards=False, trans_rewards=False, verbose=True):
            Executes PRISM to generate the game states, transitions, and other components.

        _parse_prism_files(labels=False, state_rewards=False, trans_rewards=False):
            Parses the PRISM-generated files (.sta, .tra, etc.) and builds the `GraphGame` object.

        _parse_prism_states(game):
            Parses the states file and adds states to the `GraphGame`.

        _parse_prism_transitions(game):
            Parses the transitions file and adds transitions to the `GraphGame`.

        _parse_prism_labels(game):
            Parses the labels file and assigns labels to the states in the `GraphGame`.

        _parse_prism_state_rewards(game):
            Parses the state rewards file and assigns rewards to the states in the `GraphGame`.

        _parse_prism_trans_rewards(game):
            Parses the transition rewards file and assigns rewards to the transitions in the `GraphGame`.

        _apply_state_filter(game_graph):
            Filters out states from the game graph based on the provided filter function.

        _apply_trans_filter(game_graph):
            Filters out transitions from the game graph based on the provided filter function.

        _relabel_game_graph_nodes(game_graph):
            Relabels the nodes in the game graph with sequential IDs.

        _get_prism_model_type():
            Runs PRISM to get the model type from the input PRISM file.

        _check_model_compliance(prism_model_type, game_model_type):
            Ensures that the PRISM model type is compatible with the expected game model type.
    """
    ModelTypes = [
        "mdp",  # Markov Decision Process
        "pomdp",  # Partially Observable Markov Decision Process
        "csg",  # Concurrent Stochastic Game
        "smg",  # Stochastic Multiplayer Game
        "dtmc"  # Discrete-Time Markov Chain
    ]

    def __init__(self,
                 name,
                 model_type,
                 prism_file,
                 prism_executable=pathlib.Path("C:/Program Files/prism-games-3.2.1/bin/prism_mod.bat"),
                 qualitative=None,
                 state_filter=None,
                 trans_filter=None,
                 *,
                 tmp_folder=None,
                 auto_cleanup=True,
                 force_clear=False,
                 turn_var_name=None
                 ):

        self._prism_file = prism_file
        self._qualitative = qualitative
        self._state_filter = state_filter
        self._trans_filter = trans_filter
        self._name = name
        self._turn_var_name = turn_var_name
        self._prism_executable = prism_executable
        self._model_type = model_type
        assert model_type in ModelTypes, f"Model type must be in {ModelTypes}"

        self._tmp_folder = tmp_folder if tmp_folder is not None else pathlib.Path().resolve() / "tmp"
        self._auto_cleanup = auto_cleanup

        # Create the temporary folder if it does not exist
        if self._tmp_folder.exists() and not force_clear:
            raise FileExistsError(f"Folder at {self._tmp_folder} already exists. Please clear it or use another folder.")
        elif self._tmp_folder.exists() and force_clear:
            logger.warning(f"Folder at {self._tmp_folder} already exists. Forcing clear.")
            shutil.rmtree(self._tmp_folder)
            self._tmp_folder.mkdir()
            logger.info(f"Created folder at {self._tmp_folder}")
        else:
            self._tmp_folder.mkdir()
            logger.info(f"Created folder at {self._tmp_folder}")

    def __del__(self):
        if self._auto_cleanup:
            shutil.rmtree(self._tmp_folder)
            logger.info(f"Deleted folder at {self._tmp_folder}")

    def build(self, labels=False, state_rewards=False, trans_rewards=False, verbose=False, check_compliance=False):
        # Build prism model and generate intermediate files.
        self._run_prism(labels, state_rewards, trans_rewards, verbose)

        # Get model type of input prism file.
        prism_model_type = self._get_prism_model_type()
        self._check_model_compliance(prism_model_type, self._model_type)

        # Parse generated files to construct networkx graph object.
        game_graph = self._parse_prism_files(labels, state_rewards, trans_rewards)

        # Postprocess the graph, if required.
        if self._state_filter is not None:
            game_graph = self._apply_state_filter(game_graph)

        if self._trans_filter is not None:
            game_graph = self._apply_trans_filter(game_graph)

        if self._state_filter is not None or self._trans_filter is not None:
            game_graph = self._relabel_game_graph_nodes(game_graph)

        if check_compliance:
            game_graph.check_compliance()

        # Return game graph
        return game_graph

    def _run_prism(self, labels=False, state_rewards=False, trans_rewards=False, verbose=True):
        # Run prism-games to generate the game states, transitions, etc.
        command = [
            self._prism_executable,
            self._prism_file,
            "-exportstates", str(self._tmp_folder / f"{self._name}.sta"),
            "-exporttrans", str(self._tmp_folder / f"{self._name}.tra"),
        ]

        if labels:
            command.extend([
                "-exportlabels", str(self._tmp_folder / f"{self._name}.lab"),
            ])

        if state_rewards:
            command.extend([
                "-exportstaterewards", str(self._tmp_folder / f"{self._name}.srew"),
            ])

        if trans_rewards:
            command.extend([
                "-exporttransrewards", str(self._tmp_folder / f"{self._name}.trew"),
            ])

        result = subprocess.run(command, shell=False, capture_output=True)

        if verbose:
            print("======================= PRISM Command =======================")
            print(result.args)
            print("======================= PRISM Output =======================")
            print(result.stdout.decode())

        if result.returncode != 0:
            raise RuntimeError(f"PRISM execution failed with error: {result.stdout.decode()}")

    def _parse_prism_files(self, labels=False, state_rewards=False, trans_rewards=False):
        # Create a temporary gamegraph object
        game = GraphGame(self._name, self._model_type)

        # Parse the states file
        game = self._parse_prism_states(game)

        # Parse the transitions file
        game = self._parse_prism_transitions(game)

        # Parse the labels file
        if labels:
            game = self._parse_prism_labels(game)

        # Parse the state rewards file
        if state_rewards:
            game = self._parse_prism_state_rewards(game)

        # Parse the transition rewards file
        if trans_rewards:
            game = self._parse_prism_trans_rewards(game)

        # Return the game graph
        return game

    def _parse_prism_states(self, game):
        # Extract graph
        graph = game._graph

        # Parse the states file
        with open(self._tmp_folder / f"{self._name}.sta", "r") as f:
            states = f.readlines()

        # Parse variable names
        var_names = states.pop(0)
        var_names = var_names[1:-2]  # Remove the first and last character (brackets)
        var_names = tuple(var_names.split(","))
        graph.graph["state_vars"] = var_names

        if game.is_turn_based() and self._turn_var_name is None:
            assert "turn" in var_names, "Turn-based games must have a 'turn' variable."

        # Parse the states
        while states:
            state = states.pop(0)
            state = state.strip().split(":")
            state[1] = state[1].replace("false", "False").replace("true", "True")
            graph.add_node(int(state[0]), state=ast.literal_eval(state[1]), label=set(), reward=None)

        # Update game
        game._graph = graph
        game._inv_node_map = {graph.nodes[node]["state"]: node for node in graph.nodes()}
        return game

    def _parse_prism_transitions(self, game):
        # Extract graph
        graph = game._graph

        # Parse the transitions file
        with open(self._tmp_folder / f"{self._name}.tra", "r") as f:
            transitions = f.readlines()

        transitions.pop(0)  # Remove the header
        while transitions:
            trans = transitions.pop(0)
            trans = trans.split(" ")
            if len(trans) == 5:
                src, tgt, _, prob, act = trans
            else:  # len is 4
                src, tgt, _, prob = trans
                act = None

            prob = None if self._qualitative else float(prob)
            graph.add_edge(int(src), int(tgt), prob=prob, key=act, reward=None)

        # Update game
        game._graph = graph
        return game

    def _parse_prism_labels(self, game):
        # Extract graph
        graph = game._graph

        with open(self._tmp_folder / f"{self._name}.lab", "r") as f:
            labels = f.readlines()

        # Extract atoms to construct a dictionary of atom_id to atom_name
        atoms = labels.pop(0)

        # Split the string into individual key-value pairs
        pairs = atoms.split()

        # Create a dictionary by processing each pair
        atoms = {}
        for pair in pairs:
            key, value = pair.split('=')
            atoms[int(key)] = value.strip('"')

        # Assign labels to states in the graph
        while labels:
            label = labels.pop(0)
            state, label = label.split(":")
            label = list(map(int, label.strip().split(" ")))
            label = {atoms[p] for p in label}
            if "init" in label:
                if graph.graph.get("init_states", None):
                    graph.graph["init_states"].add(int(state))
                else:
                    graph.graph["init_states"] = {int(state)}
                # graph.graph["init_states"] = graph.graph.get("init_states", set()).add(int(state))

            graph.nodes[int(state)]["label"] = set(label) - {"deadlock", "init"}

        # Update game
        game._graph = graph
        return game

    def _parse_prism_state_rewards(self, game):
        # Extract graph
        graph = game._graph

        # Collect all .srew files
        files = self._tmp_folder.rglob(f'*.srew')
        logger.debug(f"Found the following .srew files: {list(files)}")

        # Process each reward file separately
        for file in files:
            # Parse the state rewards file (Assume only one reward).
            with open(file, "r") as f:
                rewards = f.readlines()

            # Extract reward structure names
            rewards = rewards[3:]
            while rewards:
                line = rewards.pop(0)
                line = line.split(" ")

                state = int(line[0])
                rew = float(line[1])

                if "reward" in graph.nodes[state]:
                    graph.nodes[state]["reward"].update({file.stem: rew})
                else:
                    graph.nodes[state]["reward"] = {file.stem: rew}

        # Update game
        game._graph = graph
        return game

    def _parse_prism_trans_rewards(self, game):
        # Extract graph
        graph = game._graph

        # Collect all .srew files
        files = self._tmp_folder.rglob(f'*.trew')
        logger.debug(f"Found the following .trew files: {list(files)}")

        # Process each reward file separately
        for file in files:
            # Parse the transition rewards file (Assume only one reward).
            with open(file, "r") as f:
                rewards = f.readlines()

            # Extract reward structure names
            rewards = rewards[3:]
            while rewards:
                line = rewards.pop(0)
                line = line.split(" ")
                src, act, tgt, rew = int(line[0]), int(line[1]), int(line[2]), float(line[3])
                if "reward" in graph.edges[src, tgt, act]:
                    graph.nodes[src, tgt, act]["reward"].update({file.stem: rew})
                else:
                    graph.nodes[src, tgt, act]["reward"] = {file.stem: rew}

        # Update game
        game._graph = graph
        return game

    def _apply_state_filter(self, game_graph):
        rem_nodes = []
        for node in game_graph._graph.nodes():
            if not self._state_filter(game_graph._graph.nodes[node]["state"]):
                rem_nodes.append(node)
        game_graph._graph.remove_nodes_from(rem_nodes)
        return game_graph

    def _apply_trans_filter(self, game_graph):
        rem_edges = []
        for u, v, act in game_graph._graph.edges(keys=True):
            src = game_graph._graph.nodes[u]["state"]
            dst = game_graph._graph.nodes[v]["state"]
            if not self._trans_filter(src, dst, act):
                rem_edges.append((src, dst, act))
        game_graph._graph.remove_edges_from(rem_edges)
        return game_graph

    # noinspection PyMethodMayBeStatic
    def _relabel_game_graph_nodes(self, game_graph):
        # Get the underlying networkx graph
        graph = game_graph._graph

        # Create a mapping from old node IDs to new sequential IDs
        sorted_nodes = sorted(graph.nodes())  # Get sorted list of remaining nodes
        mapping = {old_id: new_id for new_id, old_id in enumerate(sorted_nodes)}

        # Relabel nodes using the new sequential IDs
        graph = nx.relabel_nodes(graph, mapping, copy=False)  # Relabel nodes in-place

        # Update the game graph object
        game_graph._graph = graph

        return game_graph

    def _get_prism_model_type(self):
        try:
            # Run prism-games to generate the game states, transitions, etc.
            command = [
                self._prism_executable,
                self._prism_file,
                "-exportprism", str(self._tmp_folder / f"{self._name}.parsedprism"),
            ]

            result = subprocess.run(command, shell=False, capture_output=True)

            if result.returncode != 0:
                raise RuntimeError()

            with open(self._tmp_folder / f"{self._name}.parsedprism", "r") as f:
                model_type = f.readline()

            return model_type.strip()

        except Exception as e:
            logger.error(f"Error while running PRISM to get model type: {e}")
            return None

    # noinspection PyMethodMayBeStatic
    def _check_model_compliance(self, prism_model_type, game_model_type):
        # Ensure prism model type is supported.
        assert prism_model_type in PrismGame.ModelTypes, f"Model type mismatch: {prism_model_type} must be in {PrismGame.ModelTypes}"

        # Ensure pairing of model types
        if game_model_type == "mdp":
            assert prism_model_type in ["mdp", "dtmc"], f"Model type mismatch: {game_model_type} must be in {['mdp', 'dtmc']}"
        elif game_model_type == "dtptb":
            assert prism_model_type in ["csg", "smg"], f"Model type mismatch: {game_model_type} must be in {['csg', 'smg']}"
        elif game_model_type == "csg":
            assert prism_model_type == "csg", f"Model type mismatch: {game_model_type} must be in {['csg']}"
        elif game_model_type == "smg":
            assert prism_model_type == "smg", f"Model type mismatch: {game_model_type} must be in {['smg']}"
        else:
            raise ValueError("Unknown model type.")


if __name__ == '__main__':
    def state_filter(state):
        turn, tom_x, tom_y, jerry_x, jerry_y = state
        if (tom_x, tom_y) in {(2, 1), (2, 2)} or (jerry_x, jerry_y) in {(2, 1), (2, 2)}:
            return True
        return False


    filename = pathlib.Path("C:/Users/Abhishek/Documents/tom_jerry.prism")
    game = PrismGame(
        name="tom_jerry",
        model_type="mdp",
        prism_file=filename,
        auto_cleanup=False,
        state_filter=state_filter
    )
    graph_game = game.build(labels=True)
