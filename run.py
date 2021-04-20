import numpy as np
from player import Player
from environment import Environment


if __name__ == "__main__":
    player = Player("Alex", [3, 4, 4, 4, 5])
    actions = player.get_actions([3, 3])
    actions = player.reorder_actions(actions)
    print(actions)
