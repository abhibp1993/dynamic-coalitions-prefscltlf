"""
Solvers for deterministic two-player turn-based game.
    * Sure winning strategy computation.
"""

import ggsolver.game as game_module
import networkx as nx
from functools import reduce
from loguru import logger


class SWinReach:
    PGSOLVER = "pgsolver"
    GGSOLVER = "ggsolver"

    def __init__(self, game: 'ggsolver.game.GraphGame', final: set, player: int = 1, solver: str = GGSOLVER, **kwargs):
        # Input parameters:
        self._game = game
        self._final = set(final)
        self._player = player

        # Solver parameters:
        self._is_solved = False
        self._solver = solver

        # Output parameters:
        self.level_set = {0: set(final)}
        self.winning_nodes = {player: set(final), 3 - player: set()}
        self.winning_edges = {player: set(), 3 - player: set()}

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

        elif self._solver == SWinReach.PGSOLVER:
            self.solve_pgsolver()

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
            logger.warning(f"Game has no final states. Marking all states to be winning for player-{3 - self._player}.")
            self.winning_nodes[3 - self._player] = self._game.states()
            self.winning_edges[3 - self._player] = self._game.transitions()
            self._is_solved = True
            return

        # Mark all final states as sink states
        graph = nx.subgraph_view(
            self._game.graph,
            filter_node=lambda n: True,
            filter_edge=lambda u, v, a: u not in self._final
        )

        # Initialization
        rank = 1
        win_nodes = set(self._final)

        while True:
            predecessors = set(reduce(set.union, map(set, map(graph.predecessors, win_nodes))))
            pre_p = {uid for uid in predecessors if graph.nodes[uid]["state"].turn() == self._player}
            pre_np = predecessors - pre_p
            pre_np = {uid for uid in pre_np if set(graph.successors(uid)).issubset(win_nodes)}
            next_level = set.union(pre_p, pre_np) - win_nodes

            if len(next_level) == 0:
                break

            self.level_set[rank] = next_level
            self.winning_edges[self._player].update(
                {(u, v, a) for u in next_level for _, v, a in graph.out_edges(u, keys=True) if v in win_nodes}
            )
            win_nodes |= next_level
            rank += 1

        # States not in win_nodes are winning for np.
        self.winning_nodes[self._player] = win_nodes
        self.winning_nodes[3 - self._player] = set(graph.nodes()) - self.winning_nodes[self._player]
        self.winning_edges[3 - self._player] = set(graph.edges(keys=True)) - self.winning_edges[self._player]

        # Mark the game to be solved
        self._is_solved = True

    def reset(self):
        self.level_set = {0: set(self._final)}
        self.winning_nodes = {self._player: set(self._final), 3 - self._player: set()}
        self.winning_edges = {self._player: set(), 3 - self._player: set()}