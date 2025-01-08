"""
Solvers for concurrent deterministic (multiplayer) game.
    * Sure winning strategy computation.
"""

import ggsolver.game as game_module
import networkx as nx
from functools import reduce
from loguru import logger
from collections import defaultdict


class SWinReach:
    GGSOLVER = "ggsolver"

    def __init__(self,
                 game: 'ggsolver.game.GraphGame',
                 final: set,
                 num_players: int,
                 player: int = 1,
                 solver: str = GGSOLVER,
                 **kwargs
                 ):
        # Input parameters:
        self._game = game
        self._final = set(final)
        self._num_player = num_players
        self._player = player

        # Solver parameters:
        self._is_solved = False
        self._solver = solver

        # Internal variables
        self.enabled_actions = self._construct_enabled_actions()  # Enabled actions for self._player from all states

        # Output parameters:
        self.level_set = {0: set(final)}
        self.winning_nodes = {player: set(final), 3 - player: set()}

    def pre(self, set_u):
        frontier = set(self._game.predecessors(set_u))

        for state in self._game.predecessors(set_u):
            eliminated_actions = set()
            for act_i in self.enabled_actions[state]:
                for _, next_state, action, _ in self._game.transitions(from_state=state):
                    if act_i == action[self._player] and next_state not in set_u:
                        eliminated_actions.add(act_i)

            if self.enabled_actions[state] == eliminated_actions:
                frontier.discard(state)

        return frontier

    def solve(self, force=False):
        # If game is solved and `force` is False, then warn the user.
        if self._is_solved and not force:
            logger.warning(f"Game is solved. To solve again, call `solve(force=True)`.")
            return

        # Invoke the appropriate solver by asserting appropriate model type.
        assert isinstance(self._game, game_module.GraphGame), \
            f"dtptb.SWinReach python solver expects model of type `GameGraph`, not `{type(self._graph)}`."

        if self._solver == SWinReach.GGSOLVER:
            self.solve_ggsolver()

        else:
            raise ValueError(f"Unknown solver: {self._solver}")

    def solve_ggsolver(self):
        """
        Expects model to be GameGraph graph.
        """
        # Reset solver
        self.reset()

        # If no final states, do not run solver. Declare all states are winning for opponent.
        if len(self._final) == 0:
            logger.warning(f"Game has no final states. Marking all states to be losing for player-{self._player}.")
            self.winning_nodes[self._player] = set()
            self._is_solved = True
            return

        # Initialization
        win_nodes = set(self._final)

        while True:
            pre = self.pre(win_nodes)
            if len(pre - win_nodes) == 0:
                break
            win_nodes |= pre

        # States not in win_nodes are winning for np.
        self.winning_nodes[self._player] = win_nodes

        # Mark the game to be solved
        self._is_solved = True

    def reset(self):
        self.level_set = {0: set(self._final)}
        self.winning_nodes = {self._player: set(self._final), 3 - self._player: set()}

    def _construct_enabled_actions(self):
        en_actions = defaultdict(set)

        # For each state in the game graph
        for state in self._game.states():
            # Iterate over all out-edges of the state
            for _, _, act, _ in self._game.transitions(from_state=state):
                # Update enabled actions dictionary
                en_actions[state].add(act[self._player])

        return en_actions
