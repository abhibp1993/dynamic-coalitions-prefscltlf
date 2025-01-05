"""
Notes.

1. Represent state as a set of atoms, i.e., ground predicates.
2. A predicate could be `on(X, Y)`, `hold(A, X)`, etc. where `X` and `Y` are blocks, and `A` is an arm.
    I think these two would be sufficient.
3. Grounding is an operation where we assign each variable in a predicate a constant from the domain.
    For example, `on(b1, b2)` and `on(b1, table)` are grounded predicates or atoms.
3. The set of boxes and arms is input to constructor of BlocksWorld class.
4. In the constructor, construct the set of grounded predicates.
    I suggest representing each grounded predicate as a tuple. E.g., `("on", "b1", "b2")`.
5. Pick an init state and implement the `init_states` method.
6. Implement the `is_state_valid` method. Think of all conditions that must hold in a valid state. For example,
    a) If `("on", "b1", "b2")` holds, then `("on", "b1", "table")` must not hold.
    b) If `("on", "b1", "b2")` holds, then `("on", bn, "b1")` must not hold for any `bn` in `boxes - {"b1", "b2"}`.
    ...
7. Implement the `actions` method. I think `pick_simple`, `pick_jenga`, `put`, and `no-op` would be sufficient.
    - Define a boolean pre-condition function for each action. You may define them as helper functions.
        For example, a preconditions for `("pick_simple", "a1", "b1")` are the following:
        a) `("on", bn, "b1")` must NOT hold for any `bn` in `boxes - {"b1"}`.
        b) `("hold", "a1", bn)` must NOT hold for any `bn` in `boxes`.
        c) ...
    - Your actions method should return all enabled actions at a given state (i.e. whose preconditions hold).
8. Implement the `delta` method. This method should return the next state given a state and an action.
    This is a post-condition function.
    - For example, if `("pick_simple", "a1", "b1")` is executed at state `s`, then the next state `s'` is obtained by
        a) Adding `("hold", "a1", "b1")` to `s`.
        b) Removing `("on", "b1", bn)` for whichever `bn` this grounded predicate is true.

Although I have explained this example for one agent, you may need to extend this to three agents.
Impose additional coalition-based restrictions and precedences for conflict resolution as necessary.

Remark. I suggest constructing a turn-based game, where P1 plays first to declare a coalition and agreed action.
    This declaration can be encoded within state as follows.

    Assuming `s` is a set of predicates, initial state is `(s, 1)` indicating that P1 plays in this state.
    Suppose that P1 selects
        a coalition `C = {1, 2}` and
        an action `a = (("pick_simple", "a1", "b1"), ("pick_jenga", "a2", "b2"))`.
    Then, the next state is `(s, a, 2)`.

    Define action(s) to process this state representation carefully.
"""

import itertools
import pickle

from pprint import pprint
from typing import Iterable, Dict, List
from ggsolver.generators import tsys
from ggsolver.generators.tsys.cls_state import *
from loguru import logger
from pathlib import Path

NUM_BLOCKS = 4
NUM_ARMS = 3

GAME_CONFIG = {
    "blocks": ['b1', 'b2', 'b3', 'b4'],
    "arms": ['a1', 'a2', 'a3'],
    "partitions": {
        "a1": {"b1"},
        "a2": {"b2"},
        "a3": {"b3", "b4"},
    },
    "priority": ['a1', 'a2', 'a3'],
}
GAME_CONFIG["name"] = f"blockworld_{len(GAME_CONFIG['blocks'])}b_{len(GAME_CONFIG['arms'])}a"

CONSTRUCTION_CONFIG = {
    "out": Path(__file__).parent / "out",
    "show_progress": True,
    "debug": False,
    "check_state_validity": True
}


class GameState(State):
    def __init__(self, predicates, turn, coalition=None, action=None):
        self._predicates = set(predicates)
        self._turn = turn
        self._coalition = coalition
        self._action = action
        super().__init__(obj=(self._predicates, self._turn, self._coalition, self._action))

    def __hash__(self):
        return hash((tuple(sorted(self._predicates)), self._turn, self._coalition, self._action))

    def __str__(self):
        if self._coalition is None:
            return f"State(p={self.predicates()}, turn={self.turn()})"
        else:
            return (f"State(p={self.predicates()}, turn={self.turn()}, "
                    f"coalition={self.coalition()}, action={self.action()})")

    def __repr__(self):
        if self._coalition is None:
            return f"State(p={self.predicates()}, turn={self.turn()})"
        else:
            return (f"State(p={self.predicates()}, turn={self.turn()}, "
                    f"coalition={self.coalition()}, action={self.action()})")

    def __eq__(self, other: 'GameState') -> bool:
        if isinstance(other, GameState):
            return (
                    self._predicates == other._predicates and
                    self._turn == other._turn and
                    self._coalition == other._coalition and
                    self._action == other._action
            )
        return False

    def get_object(self):
        return self._obj

    def predicates(self):
        return self._predicates

    def action(self):
        return self._action

    def coalition(self):
        return self._coalition

    def turn(self):
        return self._turn


class BlocksWorld(tsys.TransitionSystem):
    def __init__(self,
                 name: str,
                 blocks: Iterable[str],
                 arms: Iterable[str],
                 partitions: Dict[str, List[str]],
                 priority: List[str],
                 check_state_validity: bool = True
                 ):
        # Call base constructor
        super().__init__(
            name=name,
            model_type='dtptb',
            is_qualitative=True
        )

        # Class variables
        self.blocks = list(sorted(blocks))
        self.arms = list(sorted(arms))
        self.partitions = partitions
        self.priority = priority
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

        # Uncomment for printing
        # pprint(self.grounded_predicates)
        # print("=========== ")
        # pprint(self.grounded_actions)

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
                turn=1
            )
        }

    def actions(self, state):
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

        result = set()
        if state.turn() == 1:
            for player in range(2, len(self.arms) + 1):
                coalition = (1, player)
                for act in itertools.product(actions_arm[self.arms[0]], actions_arm[self.arms[player - 1]]):
                    result.add((coalition, act))

            return result

        else:  # state.turn() == 2:
            (_, p) = state.coalition()
            (a1, ap) = state.action()

            actions = []
            for i in range(1, len(self.arms) + 1):
                if i == 1:
                    actions.append([a1])
                    continue

                if i == p:
                    actions.append([ap])
                    continue

                actions.append(actions_arm[self.arms[i - 1]])

            return set(itertools.product(*actions)) - {tuple(("no-op",) for _ in range(len(self.arms)))}

    def delta(self, state, action):
        if state.turn() == 1:
            nstate = GameState(
                predicates=state.predicates(),
                turn=2,
                coalition=action[0],
                action=action[1]
            )

        else:
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
                turn=1
            )

        if self._check_state_validity:
            if not self._is_state_valid(nstate):
                raise ValueError(f"Invalid state: {nstate}")

        return nstate

    def delta2(self, state, action):
        if state.turn() == 1:
            nstate = GameState(
                predicates=state.predicates(),
                turn=2,
                coalition=action[0],
                action=action[1]
            )

        else:
            action = dict(zip(self.arms, action))

            # Process actions as per priority
            # pred_to_add = set()
            # pred_to_remove = set()
            new_predicates = set(state.predicates())
            for arm in self.priority:
                pred_to_add = set()
                pred_to_remove = set()
                act = action[arm]

                if act == ("no_action", arm):
                    continue


                elif act[0] == "pick":
                    block = act[2]
                    block_above, block_below, location = self._neighbors_block(new_predicates, block)

                    pred_to_add.add(("hold", act[1], block))
                    pred_to_remove.add(("on", block, block_below, location))

                    if block_above is not None:
                        pred_to_remove.add(("on", block_above, block, location))
                        pred_to_add.add(("on", block_above, block_below, location))


                else:  # act[0] == "put"
                    arm = act[1]
                    block = act[2]
                    l = act[3]
                    # on_block = act[3]
                    # pprint(act)
                    filtered_predicate = [predicate for predicate in list(new_predicates) if
                                          predicate[0] == "on" and predicate[3] == l]
                    if len(filtered_predicate) == 0:
                        pred_to_add.add(("on", block, 'table', l))
                        pred_to_remove.add(("hold", arm, block))

                    # pprint(list(new_predicates))
                    else:
                        on_block = filtered_predicate[0][1]
                        # finding the top block in location l
                        for predicate in filtered_predicate:
                            if predicate[2] == on_block:
                                on_block = predicate[1]

                        pred_to_add.add(("on", block, on_block, l))
                        pred_to_remove.add(("hold", arm, block))
                # n=new_predicates.copy()
                new_predicates -= pred_to_remove
                new_predicates |= pred_to_add
                # print("kaan")

            nstate = GameState(
                predicates=new_predicates,
                turn=1
            )

        if self._check_state_validity:
            if not self._is_state_valid(nstate):
                raise ValueError(f"Invalid state: {nstate}")

        return nstate

    def atoms(self):
        return {"a", "b", "c"}

    def label(self, state):
        if state.turn() == 1 and ('on', 'b1', 'table') in state.predicates() and ('on', 'b2', 'b1') in state.predicates():
            return {"a"}

        return set()

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
    game = BlocksWorld(
        name=GAME_CONFIG["name"],
        blocks=GAME_CONFIG["blocks"],
        arms=GAME_CONFIG["arms"],
        partitions=GAME_CONFIG["partitions"],
        priority=GAME_CONFIG["priority"],
        check_state_validity=CONSTRUCTION_CONFIG["check_state_validity"]
    )
    out = game.build(
        build_labels=True,
        show_progress=CONSTRUCTION_CONFIG["show_progress"],
        debug=CONSTRUCTION_CONFIG["debug"]
    )

    with open(CONSTRUCTION_CONFIG["out"] / f"{GAME_CONFIG['name']}.pkl", "wb") as f:
        f.write(pickle.dumps(out))
