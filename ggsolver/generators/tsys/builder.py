import multiprocessing
import pprint
import time
import textwrap
import sys

from collections.abc import Iterable
from functools import reduce
from tqdm import tqdm
from pqdm.processes import pqdm
from loguru import logger
from ggsolver import game
from ggsolver.generators.tsys.cls_state import *


class Builder:
    def __init__(self, tsys: 'TransitionSystem', **kwargs):
        # Class variables
        self.tsys = tsys
        self.game_graph = None
        self.options = {
            "pointed": kwargs.get("pointed", True),
            "show_report": kwargs.get("show_report", True),
            "build_labels": kwargs.get("build_labels", False),
            "show_progress": kwargs.get("show_progress", True),
            "multicore": kwargs.get("multicore", True),
            "debug": kwargs.get("debug", False),
        }

        # Class configuration
        if tsys.is_deterministic():
            self._get_next_states = self._get_transitions_deterministic
        elif tsys.is_probabilistic() and tsys.is_qualitative():
            self._get_next_states = self._get_transitions_qual_probabilistic
        elif tsys.is_probabilistic() and not tsys.is_qualitative():
            self._get_next_states = self._get_transitions_probabilistic
        else:
            raise ValueError(
                "Could not determine transition type of transition system. "
                "The functions `is_deterministic()`, `is_probabilistic()`, `is_qualitative()` all returned `False`."
            )

    def build(self) -> game.GraphGame:
        logger.info(f"Starting build with {self.options}.")
        start_time = time.time()
        if self.options["pointed"]:
            init_states = self._get_init_states()
            assert len(init_states) > 0, f"No initial states found for {self.tsys}."
            self.build_pointed(init_states)
        else:
            self.build_unpointed()
        end_time = time.time()

        if self.options["show_report"]:
            self.print_report(run_time=end_time - start_time)

        return self.game_graph

    def build_pointed(self, init_states) -> game.GraphGame:
        # Instantiate model
        model = game.GraphGame(name=self.tsys.name(), model_type=self.tsys.model_type())

        # Add initial states
        for state in init_states:
            assert isinstance(state, State), \
                f"Invalid state type: Expected instance of `tsys.State` got {type(state)}."
            sid = model.add_state(state)
            model.set_init_state(sid)

        # FIFO exploration
        frontier = list(model.states(as_names=True))
        frontier_set = set(frontier)
        with tqdm(total=len(frontier), disable=not self.options["show_progress"]) as pbar:
            while frontier:
                # Visit next state
                state = frontier.pop(0)
                frontier_set.remove(state)
                if self.options.get("debug", False):
                    logger.debug("State: \n" + pprint.pformat(state))

                # Get actions to explore from current state
                actions = self.tsys.actions(state)
                if self.options.get("debug", False):
                    logger.debug("Actions: \n" + pprint.pformat(actions))
                    # logger.debug(pprint.pformat(f"Actions: {actions}"))

                # Apply all actions to generate next states
                transitions = self._get_next_states(state, actions)
                for _, next_state, act_name, prob in transitions:
                    if self.options.get("debug", False):
                        logger.debug(f"Transition:\n\t{state=}\n\t{next_state=}\n\t{act_name=}\n\t{prob=}")
                    # Add next state, if not already in game graph
                    # if next_state not in (frontier_set | set(model.states(as_names=True))):
                    if next_state not in set(model.states(as_names=True)):
                        model.add_state(next_state)
                        if next_state not in frontier_set:
                            frontier.append(next_state)
                            frontier_set.add(next_state)
                    # Add transition (transition representation in GraphGame is (u, v, a))
                    model.add_transition((state, next_state, act_name, prob))

                # Update progress bar
                pbar.total = pbar.n + len(frontier)
                pbar.update(1)
                pbar.set_description(f"model.states: {model.num_states()}")
                pbar.refresh()

        # Update class variable model
        self.game_graph = model

        # Build labels
        if self.options["build_labels"]:
            self._build_labels()

    def build_unpointed(self):
        # Instantiate model
        model = game.GraphGame(name=self.tsys.name(), model_type=self.tsys.model_type())

        # Collect all states
        states = list(self.tsys.states())
        assert len(states) > 0, f"No states found in transition system."
        assert all(isinstance(state, State) for state in states), f"All states must be of type State or its derivative."

        # Collect initial states
        try:
            init_states = self.tsys.init_states()
        except NotImplementedError:
            init_states = list()
            logger.warning(f"No initial states found.")

        # Generate transitions
        if self.options["multicore"]:
            transitions = self.build_unpointed_multi_core(states)
        else:
            transitions = self.build_unpointed_single_core(states)

        # Update model with initial states
        for state in init_states:
            assert isinstance(state, State), \
                f"Invalid state type: Expected instance of `tsys.State` got {type(state)}."
            sid = model.add_state(state)
            model.set_init_state(sid)

        # Add state and transitions (Note: add_state checks for duplicates)
        model.add_states(states)
        model.add_transitions(transitions)

        # Update class variable model
        self.game_graph = model

        # Build labels
        if self.options["build_labels"]:
            self._build_labels()

    def build_unpointed_multi_core(self, states):
        cpu_count = multiprocessing.cpu_count()
        args = [
            {"state": state, "actions": self.tsys.actions(state)}
            # (state, self.tsys.actions(state))
            for state in states
        ]
        transitions = pqdm(
            args,
            self._get_next_states,
            argument_type='kwargs',
            n_jobs=cpu_count,
            desc="Generating transitions (multi-core)",
            disable=not self.options["show_progress"]
        )
        return reduce(set.union, transitions)

    def build_unpointed_single_core(self, states):
        transitions = set()
        for state in tqdm(
                states,
                desc="Generating transitions (single-core)",
                disable=not self.options["show_progress"]
        ):
            # Get actions to explore from current state
            actions = self.tsys.actions(state)

            # Generate transitions from state given actions
            transitions |= self._get_next_states(state, actions)

        return transitions

    def print_report(self, run_time):
        if self.game_graph is not None:
            labels = "Not built"
            if self.options["build_labels"]:
                atom2state = {"/no-label": 0}
                for sid, state in self.game_graph.states(as_dict=True).items():
                    label_sid = self.game_graph.get_label(state)
                    if len(label_sid) == 0:
                        atom2state["/no-label"] += 1

                    for atom in label_sid:
                        atom2state[atom] = atom2state.get(atom, 0) + 1

                labels = "\n".join(
                    " " * 16 +
                    f"- {atom}: "
                    f"{atom2state[atom]} "
                    f"states"
                    for atom in atom2state.keys()
                )
                labels = "\n" + labels + "\n"

            output = \
                f"""
                ===============================
                Model Build Report
                ===============================
                Model type: {self.game_graph.model_type()}
                Model name: {self.game_graph.name()}
                State components: {tuple(name for name in self.tsys.state_vars())}
                States: {self.game_graph.num_states()}
                Actions: {len(self.game_graph.actions())}
                Transitions: {self.game_graph.num_transitions()}
                Initial states: {len(self.game_graph.init_states())}
                Labels: {labels}
                Time taken: {run_time:.6f} seconds
                Memory used: {sys.getsizeof(self.game_graph)} bytes
                ===============================

                """

            print(textwrap.dedent(textwrap.dedent(output)))

    def _build_labels(self):
        for st in self.game_graph.states(as_names=True):
            label_sid = self.tsys.label(st)
            assert all(isinstance(p, str) for p in label_sid), "Labels must all be strings."
            self.game_graph.set_label(st, label_sid)

    def _get_transitions_deterministic(self, state, actions):
        transitions = set()
        for action in actions:
            n_state = self.tsys.delta(state, action)
            assert isinstance(n_state, State), \
                (f"Invalid state type: Expected instance of `tsys.State` got {type(n_state)}.\n"
                 f"Input: {state=}, {action=}")
            transitions.add((state, n_state, action, None))  # (action, next_state, prob)

        return transitions

    def _get_transitions_qual_probabilistic(self, state, actions):
        transitions = set()
        for action in actions:
            n_state_list = self.tsys.delta(state, action)
            assert isinstance(n_state_list, Iterable), (f"Qualitative probabilistic transition function must return "
                                                        f"an iterable of `tsys.State` or derived objects.")
            for n_state in n_state_list:
                assert isinstance(n_state, State), \
                    f"Invalid state type: Expected instance of `tsys.State` got {type(n_state)}."
                transitions.add((state, n_state, action, None))  # (action, next_state, prob)
        return transitions

    def _get_transitions_probabilistic(self, state, actions):
        transitions = set()
        for action in actions:
            n_state_dict = self.tsys.delta(state, action)
            assert isinstance(n_state_dict, dict), f"Probabilistic transition function must return a dictionary."
            for n_state, prob in n_state_dict.items():
                assert isinstance(n_state, State), \
                    f"Invalid state type: Expected instance of `tsys.State` got {type(n_state)}."
                transitions.add((state, n_state, action, prob))  # (action, next_state, prob)

        return transitions

    def _get_init_states(self):
        try:
            return self.tsys.init_states()
        except NotImplementedError:
            logger.warning(f"{self.tsys}.init_states() is not implemented. "
                           f"Using {self.tsys}.states() to determine initial states.")

        return self.tsys.states()
