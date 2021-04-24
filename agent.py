import numpy as np
from player import Player


class Agent(Player):
    def __init__(self, name, hand, rank='Person'):
        super().__init__(name, hand, rank)
    
    def play_action(self, active_card):
        pass
        