import itertools
import multiprocessing
import sys
import textwrap
import time

from collections.abc import Iterable
from functools import reduce
from loguru import logger
from tqdm import tqdm
from pqdm.processes import pqdm
from ggsolver.game import GraphGame
from ggsolver.generators.modules.cls_state import *


class Builder:
    def __init__(self, module: 'Module', **kwargs):
        # Class variables
        self.module = module
        self.game_graph = None
        self.options = {
            "pointed": kwargs.get("pointed", True),
            "show_report": kwargs.get("show_report", True),
            "build_labels": kwargs.get("build_labels", False),
            "show_progress": kwargs.get("show_progress", True),
            "multicore": kwargs.get("multicore", True)
        }

        # Class configuration
        if module.is_deterministic():
            self._get_transitions = self._get_transitions_deterministic
        elif module.is_probabilistic() and module.is_qualitative():
            self._get_transitions = self._get_transitions_qual_probabilistic
        elif module.is_probabilistic() and not module.is_qualitative():
            self._get_transitions = self._get_transitions_probabilistic
        else:
            raise ValueError(
                "Could not determine transition type of transition system. "
                "The functions `is_deterministic()`, `is_probabilistic()`, `is_qualitative()` all returned `False`."
            )

    def build(self):

        start_time = time.time()
        if self.options["pointed"]:
            assert len(self.module.init_states()) > 0, "No initial state provided."
            self.build_pointed()
        else:
            self.build_unpointed()
        end_time = time.time()

        if self.options["show_report"]:
            self.print_report(run_time=end_time - start_time)

    def build_pointed(self):
        # Instantiate model
        model = GraphGame(name=self.module.name(), model_type=self.module.model_type())

        # Add initial states
        for state in self.module.init_states():
            sid = model.add_state(state)
            model.set_init_state(sid)

        # FIFO exploration
        frontier = list(model.states(as_dict=True).values())
        frontier_set = set(frontier)

        with tqdm(total=len(frontier), disable=not self.options["show_progress"], desc="Generating states") as pbar:
            while frontier:
                # Visit next state
                state = frontier.pop(0)
                frontier_set.remove(state)

                # Get actions to explore from current state
                actions = self.module.enabled_actions(state)

                # Apply all actions to generate next states
                next_states = self._get_transitions(state, actions)
                for state, next_state, act_name, prob in next_states:
                    if next_state not in frontier_set | set(model.states(as_dict=True).values()):
                        model.add_state(next_state)
                        frontier.append(next_state)
                        frontier_set.add(next_state)
                    model.add_transition((state, next_state, act_name, next_state))

                # pbar.update(1)
                pbar.total = pbar.n + len(frontier)
                pbar.update(1)
                pbar.refresh()

        self.game_graph = model

        # Assign labels
        if self.options["build_labels"]:
            self._build_labels()

    def build_unpointed(self):
        # Instantiate model
        model = GraphGame(name=self.module.name(), model_type=self.module.model_type())

        # Collect all states
        ordered_name_domain_pairs = self.module.state_vars().items()
        var_names = [pair[0] for pair in ordered_name_domain_pairs]
        domains = [pair[1] for pair in ordered_name_domain_pairs]
        states = [State(**{n: val for n, val in zip(var_names, val)}) for val in itertools.product(*domains)]
        assert len(states) > 0, f"No states found in transition system."

        # Collect initial states
        try:
            init_states = self.module.init_states()
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
            {"state": state, "actions": self.module.enabled_actions(state)}
            # (state, self.tsys.actions(state))
            for state in states
        ]
        transitions = pqdm(
            args,
            self._get_transitions,
            argument_type='kwargs',
            n_jobs=cpu_count,
            desc="Generating transitions (multi-core)",
            disable=not self.options["show_progress"]
        )
        return reduce(set.union, map(set, transitions))

    def build_unpointed_single_core(self, states):
        transitions = set()
        for state in tqdm(
                states,
                desc="Generating transitions (single-core)",
                disable=not self.options["show_progress"]
        ):
            # Get actions to explore from current state
            actions = self.module.enabled_actions(state)

            # Generate transitions from state given actions
            transitions |= set(self._get_transitions(state, actions))

        return transitions

    def print_report(self, run_time):
        if self.game_graph is not None:
            labels = None
            if self.options["build_labels"]:
                labels = "\n".join(
                    " " * 16 + f"- {atom}: {len([st for st in self.game_graph.atoms()[atom] if self.game_graph.atoms()[atom][st]])} states"
                    for atom in self.game_graph.atoms()
                )
                labels = "\n" + labels + "\n"

            output = \
                f"""
                ===============================
                Model Build Report
                ===============================
                Model type: {self.game_graph.model_type()}
                Model name: {self.game_graph.name()}
                State components: {tuple(name for name in self.module.state_vars())}
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

    def _get_transitions_deterministic(self, state, actions):
        try:
            next_states = []
            for act, func in actions.items():
                # Get next state
                next_state = func(state)

                # Ensure: User-defined state validity
                if self._is_state_valid(next_state):
                    # Add the transition to next_states
                    next_states.append((state, next_state, act, None))

            return next_states
        except Exception as err:
            print(err)
            return []

    def _get_transitions_qual_probabilistic(self, state, actions):
        transitions = set()
        for act, func in actions.items():
            n_state_list = func(state)
            assert isinstance(n_state_list, Iterable), (f"Qualitative probabilistic transition function must return "
                                                        f"an iterable of `tsys.State` or derived objects.")
            for n_state in n_state_list:
                assert isinstance(n_state, State), \
                    f"Invalid state type: Expected instance of `tsys.State` got {type(n_state)}."
                transitions.add((state, n_state, act, None))  # (action, next_state, prob)
        return transitions

    def _get_transitions_probabilistic(self, state, actions):
        transitions = set()
        for act, func in actions.items():
            n_state_dict = func(state)
            assert isinstance(n_state_dict, dict), f"Probabilistic transition function must return a dictionary."
            for n_state, prob in n_state_dict.items():
                assert isinstance(n_state, State), \
                    f"Invalid state type: Expected instance of `tsys.State` got {type(n_state)}."
                transitions.add((state, n_state, act, prob))  # (action, next_state, prob)

        return transitions

    def _is_state_valid(self, state):
        # Ensure: Action function returned a state object
        if not isinstance(state, State):  # f"Action {act}({state=}) did not return a modules.State object. "
            return False

        # Ensure: State components match module state definition
        state_vars = state.vars()
        if set(self.module.state_vars()) != set(state_vars):
            return False
            # (f"Action {act}({state=}) returned a state whose components {state_vars} do not match "
            #  f"module definition {self.module.state_vars()}")

        # Ensure: Each variable value is within its domain.
        for varname, domain in self.module.state_vars().items():
            # print(varname, domain, state.__getattribute__(varname))
            if state.__getattribute__(varname) not in domain:
                # logger.warning(
                #     f"Action {act}({state=}) produced state {next_state} whose "
                #     f"component '{varname}' is out of domain."
                # )
                return False

        return True

    def _build_labels(self):
        for atom, func in self.module.atoms().items():
            label_dict = dict()
            for sid, st in self.game_graph.states(as_dict=True).items():
                label_dict[sid] = func(st)
                assert isinstance(label_dict[sid], bool), f"Label {atom} for {sid} is not Boolean."
            self.game_graph.update_labels(atom, label_dict)
