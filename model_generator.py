"""
The code generates game model and stores it in corresponding example folder.
"""

import pickle
from ggsolver.generators.tsys.cls_state import *
from pathlib import Path

# ======================================================================================================================
# MODIFY ONLY THIS BLOCK
# ======================================================================================================================
EXAMPLE = "example4"  # Folder name of your blocks world implementation
from example4.game_model import BlocksWorld

GAME_CONFIG = {
    "name": f"blockworld_4b_3a_3l",      # Name is generated automatically. If you want custom name, uncomment this line.
    "blocks": ['b1', 'b2', 'b3', 'b4'],
    "arms": ['a1', 'a2', 'a3'],
    "partitions": {
        "a1": {"b1"},
        "a2": {"b2"},
        "a3": {"b3", "b4"},
    },
    "priority": ['a1', 'a2', 'a3'],
    "specs": {      # File type `.prefscltl` will be added automatically.
        "a1": "spec1",
        "a2": "spec2",
        "a3": "spec3",
    },
    "location": 3
}

CONSTRUCTION_CONFIG = {
    "out": Path(__file__).parent / EXAMPLE / "out",
    "show_progress": True,
    "debug": False,
    "check_state_validity": True
}

if not CONSTRUCTION_CONFIG["out"].exists():
    CONSTRUCTION_CONFIG["out"].mkdir(parents=False)

name = f"blockworld_{len(GAME_CONFIG['blocks'])}b_{len(GAME_CONFIG['arms'])}a"

if (CONSTRUCTION_CONFIG["out"] / f"{name}.conf").exists():
    raise FileExistsError(f"Game model with name '{name}' already exists. Please change the name in GAME_CONFIG.")

GAME_CONFIG["name"] = name


# ======================================================================================================================


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


if __name__ == '__main__':
    game = BlocksWorld(
        name=GAME_CONFIG["name"],
        blocks=GAME_CONFIG["blocks"],
        arms=GAME_CONFIG["arms"],
        partitions=GAME_CONFIG["partitions"],
        priority=GAME_CONFIG["priority"],
        location=GAME_CONFIG["location"]
        # check_state_validity=CONSTRUCTION_CONFIG["check_state_validity"]
    )
    out = game.build(
        build_labels=True,
        show_progress=CONSTRUCTION_CONFIG["show_progress"],
        debug=CONSTRUCTION_CONFIG["debug"]
    )

    # Save game model as pickle file
    out_dir = CONSTRUCTION_CONFIG["out"]

    with open(out_dir / f"{GAME_CONFIG['name']}.conf", "wb") as f:
        f.write(pickle.dumps(GAME_CONFIG))

    with open(out_dir / f"{GAME_CONFIG['name']}.pkl", "wb") as f:
        f.write(pickle.dumps(out))
