import pickle
from ggsolver.solvers.dtptb import SWinReach
from pathlib import Path
from product import ProductState
from pprint import pprint

# ======================================================================================================================
# MODIFY ONLY THIS BLOCK
# ======================================================================================================================
EXAMPLE = "example2"  # Folder nPrefAutomatoname of your blocks world implementation
GAME_CONFIG_FILE = "blockworld_4b_3a.conf"


if __name__ == '__main__':
# Load game config
    with open(Path(__file__).parent / EXAMPLE / "out" / GAME_CONFIG_FILE, "rb") as f:
        game_config = pickle.load(f)

    # Load product game
    with open(Path(__file__).parent / EXAMPLE / "out" / f"{game_config['name']}_product.pkl", "rb") as f:
        product_game = pickle.loads(f.read())

    # Load ranks
    with open(Path(__file__).parent / EXAMPLE / "out" / f"{game_config['name']}_ranks.pkl", "rb") as f:
        ranks = pickle.loads(f.read())

    for states

    s=SWinReach(product_game, 1, GGSOLVER)

    solve_ggsolve


