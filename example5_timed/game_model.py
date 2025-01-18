"""
Game model with limited time.
"""

import itertools
import pickle
import pprint
from pathlib import Path
from typing import Iterable, Dict, List

from loguru import logger

from ggsolver.generators import tsys
from ggsolver.generators.tsys.cls_state import *


class GameState(State):
    def __init__(self, predicates, clock):
        self._predicates = set(predicates)
        self._clock = clock
        super().__init__(obj=(self._predicates, self._clock))

    def __hash__(self):
        return hash((tuple(sorted(self._predicates)), self._clock))

    def __str__(self):
        return f"State(p={self.predicates()}, clk={self.clock()})"

    def __repr__(self):
        return f"State(p={self.predicates()}, clk={self.clock()})"

    def __eq__(self, other: 'GameState') -> bool:
        if isinstance(other, GameState):
            return (
                    self._predicates == other._predicates and
                    self._clock == other._clock
            )
        return False

    def get_object(self):
        return self._obj

    def predicates(self):
        return self._predicates

    def clock(self):
        return self._clock


class BlocksWorld(tsys.TransitionSystem):
    def __init__(self,
                 name: str,
                 blocks: Iterable[str],
                 arms: Iterable[str],
                 partitions: Dict[str, List[str]],
                 priority: List[str],
                 max_time: int,
                 check_state_validity: bool = True,
                 ):
        # Call base constructor
        super().__init__(
            name=name,
            model_type='cdg',
            is_qualitative=True
        )

        # Class variables
        self.blocks = list(sorted(blocks))
        self.arms = list(sorted(arms))
        self.partitions = partitions
        self.priority = priority
        self.max_time = max_time
        self._check_state_validity = check_state_validity

        # If user has added "table" as a block, discard it
        if "table" in self.blocks:
            self.blocks.remove("table")

        # Ground predicates
        blocks_with_table = self.blocks + ["table"]
        self.grounded_predicates = [
                                       ("on", block_top, block_below)
                                       for block_top, block_below in itertools.product(self.blocks, blocks_with_table)
                                       if block_top != block_below
                                   ] + [
                                       ("hold", arm, block)
                                       for arm, block in itertools.product(self.arms, self.blocks)
                                   ]

        # Ground actions
        self.grounded_actions = [
                                    ("pick", arm, block)
                                    for arm in self.arms
                                    for block in self.blocks
                                    if block in self.partitions[arm]
                                ] + [
                                    ("put", arm, block, on_block)
                                    for arm in self.arms
                                    for block, on_block in itertools.product(self.blocks, blocks_with_table)
                                    if block in self.partitions[arm] and block != on_block
                                ]

        logger.info("Predicates in game:\n" + pprint.pformat(self.grounded_predicates))
        logger.info("Actions in game:\n" + pprint.pformat(self.grounded_actions))

    def state_vars(self):
        return ["_".join(pred) for pred in self.grounded_predicates]

    def states(self):
        return {
            GameState(
                predicates={
                               ('on', self.blocks[0], 'table')
                           } | {
                               ('on', self.blocks[i + 1], self.blocks[i]) for i in range(len(self.blocks) - 1)
                           },
                clock=0
            )
        }

    def actions(self, state):
        # No actions when clock is maxed out (aimed at inducing a self-loop)
        if state.clock() >= self.max_time:
            return set()
            # return {("no-op", "no-op", "no-op")}

        # Identify blocks on top of stack and free arms
        top_blocks = self._top_blocks(state.predicates())
        free_arms, block_held_by = self._free_arms(state)
        # print(top_blocks)
        # print(free_arms, block_held_by)

        # Every free arm can pick a block not held by some arm
        actions_arm = {arm: {("no-op",)} for arm in self.arms}
        for arm in free_arms:
            for block in set(self.partitions[arm]) - free_arms:
                actions_arm[arm].add(("pick", arm, block))

        # Every holding arm can put the block it holds on a top block
        for arm in [a for a in self.arms if a not in free_arms]:
            for block in top_blocks:
                actions_arm[arm].add(("put", arm, block_held_by[arm], block))

        return set(itertools.product(*actions_arm.values())) - {tuple(("no-op",) for _ in range(len(self.arms)))}

    def delta(self, state, action):
        if state.clock() > self.max_time:
            raise ValueError(f"Invalid state. {state.clock()} is greater than {self.max_time}")

        action = dict(zip(self.arms, action))

        # Process actions as per priority
        new_predicates = set(state.predicates())
        for arm in self.priority:
            pred_to_add = set()
            pred_to_remove = set()

            act = action[arm]

            if act == ("no-op",):
                continue

            elif act[0] == "pick":
                block = act[2]
                block_above, block_below = self._neighbors_block(new_predicates, block)

                pred_to_add.add(("hold", act[1], block))
                pred_to_remove.add(("on", block, block_below))

                if block_above is not None:
                    pred_to_remove.add(("on", block_above, block))
                    pred_to_add.add(("on", block_above, block_below))

            else:  # act[0] == "put"
                arm = act[1]
                block = act[2]
                on_block = act[3]

                if on_block != "table" and on_block not in self._top_blocks(new_predicates):
                    continue

                pred_to_add.add(("on", block, on_block))
                pred_to_remove.add(("hold", arm, block))

            new_predicates -= pred_to_remove
            new_predicates |= pred_to_add

        nstate = GameState(
            predicates=new_predicates,
            clock=state.clock() + 1
        )

        if self._check_state_validity:
            if not self._is_state_valid(nstate):
                raise ValueError(f"Invalid state: {nstate}")

        return nstate

    def atoms(self):
        return {"a", "b", "c"}

    def label(self, state):
        if ('on', 'b1', 'b2') in state.predicates():
            return {"a"}
        else:
            return {"b", "c"}
        # else:
        #     return set()

    def _free_arms(self, state):
        free_arms = set(self.arms)
        blocks_in_arms = {arm: None for arm in self.arms}

        for pred in state.predicates():
            if pred[0] == "hold":
                free_arms.discard(pred[1])
                blocks_in_arms[pred[1]] = pred[2]
        return free_arms, blocks_in_arms

    def _top_blocks(self, predicates):
        free_blocks = set(self.blocks)
        for pred in predicates:
            if pred[0] == "on":
                free_blocks.discard(pred[2])
            if pred[0] == "hold":
                free_blocks.discard(pred[2])
        return free_blocks

    def _neighbors_block(self, predicates, block):
        block_above = None
        block_below = None
        for pred in predicates:
            if pred[0] == "on" and pred[2] == block:
                block_above = pred[1]

            if pred[0] == "on" and pred[1] == block:
                block_below = pred[2]

        return block_above, block_below

    def _is_state_valid(self, state):
        if not self._chk_one_block_on_two(state):
            logger.warning(f"Invalid {state=}: One block is on two blocks.")
            return False

        if not self._chk_two_blocks_on_one(state):
            logger.warning(f"Invalid {state=}: Two blocks are on one block.")
            return False

        if not self._chk_block_on_itself(state):
            logger.warning(f"Invalid {state=}: A block is on itself.")
            return False

        if not self._chk_block_noton_something_and_in_arm(state):
            logger.warning(f"Invalid {state=}: A block is on something and in an arm.")
            return False

        if not self._chk_not_block_in_arm_and_something_on_block(state):
            logger.warning(f"Invalid {state=}: A block is on something and in an arm.")
            return False

        if not self._chk_block_is_somewhere(state):
            logger.warning(f"Invalid {state=}: A block is neither on another block nor in an arm.")
            return False

        if not self._chk_block_notin_two_arms(state):
            logger.warning(f"Invalid {state=}: A block is in two arms.")
            return False

        if not self._chk_not_table_on_block(state):
            logger.warning(f"Invalid {state=}: Table is on a block.")
            return False

        if not self._chk_not_arm_holds_table(state):
            logger.warning(f"Invalid {state=}: An arm holds the table.")
            return False

        return True

    def _chk_one_block_on_two(self, state):
        """ Constructs a dictionary {x: y} where x is the block on top and y is the block below. """
        below = dict()
        for pred, x, y in state.predicates():
            if pred == "on":
                if x in below:
                    return False
                below[x] = y
        return True

    def _chk_two_blocks_on_one(self, state):
        """ Constructs a dictionary {x: y} where x is the block below and y is the block on top. """
        above = dict()
        for pred, x, y in state.predicates():
            if pred == "on":
                if y in above:
                    return False
                above[y] = x
        return True

    def _chk_block_on_itself(self, state):
        for pred, x, y in state.predicates():
            if pred == "on" and x == y:
                return False
        return True

    def _chk_block_noton_something_and_in_arm(self, state):
        # Assume. _chk_one_block_on_two and _chk_two_blocks_on_one and _chk_block_on_itself are passed.
        position = dict()
        for pred, x, y in state.predicates():
            if pred == "on":
                if x in position:
                    return False
                position[x] = "on"
            elif pred == "hold":
                if y in position:
                    return False
                position[y] = "hold"
        return True

    def _chk_not_block_in_arm_and_something_on_block(self, state):
        # Assume. _chk_one_block_on_two and _chk_two_blocks_on_one and _chk_block_on_itself are passed.
        position = dict()
        for pred, x, y in state.predicates():
            if pred == "on":
                if y in position:
                    return False
                position[y] = "on"
            elif pred == "hold":
                if y in position:
                    return False
                position[y] = "hold"
        return True

    def _chk_block_is_somewhere(self, state):
        for block in self.blocks:

            if not any(
                    (pred[0] == "on" and pred[1] == block) or (pred[0] == "hold" and pred[2] == block)
                    for pred in state.predicates()
            ):
                return False
        return True

    def _chk_block_notin_two_arms(self, state):
        for block in self.blocks:
            if len([pred for pred in state.predicates() if pred[0] == "hold" and pred[2] == block]) > 1:
                return False
        return True

    def _chk_not_table_on_block(self, state):
        return not any(pred[0] == "on" and pred[1] == "table" for pred in state.predicates())

    def _chk_not_arm_holds_table(self, state):
        return not any(pred[0] == "hold" and pred[2] == "table" for pred in state.predicates())


if __name__ == '__main__':
    blocks = ['b1', 'b2', 'b3', 'b4']
    arms = ['a1', 'a2', 'a3']
    partitions = {
        "a1": {"b1"},
        "a2": {"b2"},
        "a3": {"b3", "b4"},
    }

    game = BlocksWorld(
        name="BW_5b_3a",
        blocks=blocks,
        arms=arms,
        partitions=partitions,
        max_time=10,
        priority=arms
    )
    game_graph = game.build(build_labels=True, show_progress=True, debug=False)
    game_graph.save_states(Path().parent / 'out' / 'example5_timed_graph.sta')
    game_graph.save_transitions(Path().parent / 'out' / 'example5_timed_graph.tra', state_names=True)

    with open("game_model.pickle", "wb") as f:
        f.write(pickle.dumps(game_graph))

    with open("game_model.pickle", "rb") as f:
        out = pickle.loads(f.read())
