"""
And Kaan implementation, 4 Jan @ 12.00 PM
"""

import itertools
import pickle

from pprint import pprint
from typing import Iterable, Dict, List
from ggsolver.generators import tsys
from ggsolver.generators.tsys.cls_state import *
from loguru import logger


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
                 # check_state_validity: bool = True,
                 location: int,
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
        # self._check_state_validity = check_state_validity
        self.location = location

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

        self.grounded_actions = ([
                                     ("pick", arm, block)
                                     for arm in self.arms
                                     for block in self.blocks
                                     if block in self.partitions[arm]
                                 ] +
                                 [
                                     ("put", arm, block, l)
                                     for arm in self.arms
                                     for l in range(self.location)
                                     for block in self.partitions[arm]
                                 ])

        #  self.grounded_actions = [
        #     ("pick", arm, block)
        #     for arm in self.arms
        #     for block in self.blocks
        #     if block in self.partitions[arm]
        # ] + [
        #     ("put", arm, block, on_block)
        #     for arm in self.arms
        #     for block, on_block in itertools.product(self.blocks, blocks_with_table)
        #     if block in self.partitions[arm] and block != on_block
        # ]

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
                    ('on', 'b1', 'table', 0),
                    ('on', 'b2', 'b1', 0),
                    ('on', 'b3', 'b2', 0),
                    ('on', 'b4', 'b3', 0),
                },
                turn=2
            )
        }

        # return {
        #     GameState(
        #         predicates={
        #                        ('on', self.blocks[0], 'table')
        #                    } | {
        #                        ('on', self.blocks[i + 1], self.blocks[i]) for i in range(len(self.blocks) - 1)
        #                    },
        #         turn=1
        #     )
        # }

    def actions(self, state):
        # state=list(state)
        available_actions = {arm: {('no_action', arm)} for arm in self.arms}

        for b1 in self.blocks:
            for b2 in self.blocks:
                for l in range(self.location):
                    if ('on', b1, b2, l) in state.predicates() or ('on', b2, b1, l) in state.predicates():
                        for arm in self.arms:
                            if b1 in self.partitions[arm] and ('hold', arm, 'none') in state.predicates():
                                if ('pick', arm, b1, l) not in available_actions[arm]:
                                    available_actions[arm].add(('pick', arm, b1, l))
            for arm in self.arms:
                if ('hold', arm, b1) in state.predicates():
                    for l in range(self.location):
                        available_actions[arm].add(('put', arm, b1, l))

    def delta(self, state, action):
        # state=list(state)
        act = self.actions(state)
        state_up = state.copy()

        a = list(act.keys())
        for i in range(len(a)):
            if action[i] not in act[a[i]]:
                print('Not a valid action profile')
                return

        for i in range(len(a)):
            if action[i] == 'no_action':
                next_state = state_up
            elif action[i][0] == 'put':
                loc = action[i][3]
                # print(state)
                filtered_predicate = [predicate for predicate in state_up if
                                      predicate[-1] == loc and predicate[0] == 'on']
                # arm tries to put it on the table, not on an existing box
                if len(filtered_predicate) == 0:
                    # clear the arm
                    state_up.discard(('hold', action[i][1], action[i][2]))
                    state_up.add(('hold', action[i][1], 'none'))
                    # put the box on the table
                    state_up.add(('on', action[i][2], 'table', loc))
                    next_state = state_up
                else:
                    b1_values = [predicate[1] for predicate in filtered_predicate]
                    b2_values = [predicate[2] for predicate in filtered_predicate]
                    unique_b1 = [b1 for b1 in b1_values if b1_values.count(b1) == 1 and b1 not in b2_values]
                    unique_b1 = unique_b1[0]
                    # put action[i][2] onto the top of loc
                    state_up.add(('on', action[i][2], unique_b1, loc))
                    # clear the arm
                    state_up.discard(('hold', action[i][1], action[i][2]))
                    state_up.add(('hold', action[i][1], 'none'))
                    next_state = state_up

            elif action[i][0] == 'pick':
                filtered_predicate = [predicate for predicate in state_up if
                                      predicate[1] == action[i][2] or predicate[2] == action[i][2]]
                # print(filtered_predicate)
                # Arm is pulling from top of a stack
                if len(filtered_predicate) == 1:
                    state_up.discard(('hold', action[i][1], 'none'))
                    state_up.add(('hold', action[i][1], action[i][2]))
                    state_up.discard(filtered_predicate[0])
                    next_state = state_up
                # Arm is pulling from the middle of a stack, i.e., jenga-move
                if len(filtered_predicate) == 2:
                    loc = filtered_predicate[1][3]
                    # print(filtered_predicate)
                    for j in range(len(filtered_predicate)):

                        if filtered_predicate[j][1] == action[i][2]:
                            b1 = filtered_predicate[j][2]

                        if filtered_predicate[j][2] == action[i][2]:
                            b2 = filtered_predicate[j][1]

                    state_up.discard(('hold', action[i][1], 'none'))
                    state_up.add(('hold', action[i][1], action[i][2]))
                    state_up.discard(('on', b2, action[i][2], loc))
                    # print(state)
                    # print(('on', action[i][2],b1, loc))
                    state_up.discard(('on', action[i][2], b1, loc))
                    state_up.add(('on', b2, b1, loc))
                    next_state = state_up

        return GameState(
                predicates=next_state,
                turn=2
            )

# def actions(self, state):
#     # Identify blocks on top of stack and free arms
#     # top_blocks = self._top_blocks(state)
#     free_arms, block_held_by = self._free_arms(state)
#     # print(top_blocks)
#     # print(free_arms, block_held_by)
#
#     # Every free arm can pick a block not held by some arm
#     actions_arm = {arm: {("no_action", arm)} for arm in self.arms}
#     for arm in free_arms:
#         for block in set(self.partitions[arm]) - free_arms:
#             actions_arm[arm].add(("pick", arm, block))
#
#     # Every holding arm can put the block it holds on a top block
#     for arm in [a for a in self.arms if a not in free_arms]:
#         for l in range(self.location):
#             actions_arm[arm].add(("put", arm, block_held_by[arm], l))
#
#     # for arm in [a for a in self.arms if a not in free_arms]:
#     #     for block in top_blocks:
#     #         actions_arm[arm].add(("put", arm, block_held_by[arm], block))
#
#     result = set()
#     if state.turn() == 1:
#         for act in actions_arm[self.arms[0]]:
#             coalition = 1
#             result.add((coalition, act))
#
#         for player in range(2, len(self.arms) + 1):
#             coalition = (1, player)
#             for act in itertools.product(actions_arm[self.arms[0]], actions_arm[self.arms[player - 1]]):
#                 result.add((coalition, act))
#
#         return result
#
#     else:  # state.turn() == 2:
#         if state.coalition() == 1:
#             a1 = state.action()
#             actions = []
#             for i in range(1, len(self.arms) + 1):
#                 if i == 1:
#                     actions.append([a1])
#                     continue
#
#                 actions.append(actions_arm[self.arms[i - 1]])
#
#             return set(itertools.product(*actions))
#
#         if state.coalition() != 1:
#             (_, p) = state.coalition()
#             (a1, ap) = state.action()
#
#             actions = []
#             for i in range(1, len(self.arms) + 1):
#                 if i == 1:
#                     actions.append([a1])
#                     continue
#
#                 if i == p:
#                     actions.append([ap])
#                     continue
#
#                 actions.append(actions_arm[self.arms[i - 1]])
#
#             return set(itertools.product(*actions))
#
#         # return set(itertools.product(*actions)) - {tuple(("no-op",) for _ in range(len(self.arms)))}
#
#
# def delta(self, state, action):
#     if state.turn() == 1:
#         return GameState(
#             predicates=state.predicates(),
#             turn=2,
#             coalition=action[0],
#             action=action[1]
#         )
#
#     else:
#         action = dict(zip(self.arms, action))
#
#         # Process actions as per priority
#         # pred_to_add = set()
#         # pred_to_remove = set()
#         new_predicates = set(state.predicates())
#         for arm in self.priority:
#             pred_to_add = set()
#             pred_to_remove = set()
#             act = action[arm]
#
#             if act == ("no_action", arm):
#                 continue
#
#
#             elif act[0] == "pick":
#                 block = act[2]
#                 block_above, block_below, location = self._neighbors_block(new_predicates, block)
#
#                 pred_to_add.add(("hold", act[1], block))
#                 pred_to_remove.add(("on", block, block_below, location))
#
#                 if block_above is not None:
#                     pred_to_remove.add(("on", block_above, block, location))
#                     pred_to_add.add(("on", block_above, block_below, location))
#
#
#             else:  # act[0] == "put"
#                 arm = act[1]
#                 block = act[2]
#                 l = act[3]
#                 # on_block = act[3]
#                 # pprint(act)
#                 filtered_predicate = [predicate for predicate in list(new_predicates) if
#                                       predicate[0] == "on" and predicate[3] == l]
#                 if len(filtered_predicate) == 0:
#                     pred_to_add.add(("on", block, 'table', l))
#                     pred_to_remove.add(("hold", arm, block))
#
#                 # pprint(list(new_predicates))
#                 else:
#                     on_block = filtered_predicate[0][1]
#                     # finding the top block in location l
#                     for predicate in filtered_predicate:
#                         if predicate[2] == on_block:
#                             on_block = predicate[1]
#
#                     pred_to_add.add(("on", block, on_block, l))
#                     pred_to_remove.add(("hold", arm, block))
#             # n=new_predicates.copy()
#             new_predicates -= pred_to_remove
#             new_predicates |= pred_to_add
#             # print("kaan")
#
#         return GameState(
#             predicates=new_predicates,
#             turn=1
#         )


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
    location = None
    for pred in predicates:
        if pred[0] == "on" and pred[2] == block:
            block_above = pred[1]

        if pred[0] == "on" and pred[1] == block:
            block_below = pred[2]
            location = pred[3]

    return block_above, block_below, location


# def _neighbors_block(self, predicates, block):
#     block_above = None
#     block_below = None
#     for pred in predicates:
#         if pred[0] == "on" and pred[2] == block:
#             block_above = pred[1]
#
#         if pred[0] == "on" and pred[1] == block:
#             block_below = pred[2]
#
#     return block_above, block_below

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

    game = BlocksWorld(name="BW_5b_3a", blocks=blocks, arms=arms, partitions=partitions, priority=arms, location=2)
    out = game.build(build_labels=True, show_progress=True, debug=False)
    with open("game_model.pickle", "wb") as f:
        f.write(pickle.dumps(out))

    with open("game_model.pickle", "rb") as f:
        out = pickle.loads(f.read())
