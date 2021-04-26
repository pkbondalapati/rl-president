import numpy as np
from player import Player


class Agent(Player):
    def __init__(self, max_cards, name='Agent', hand=[], rank='Person'):
        super().__init__(name, hand, rank)
        self.max_cards = max_cards
    
    # Get all active cards, given 'n' card count.
    def get_cards(self, n):
        unique_cards = np.arange(3, self.max_cards)
        cards = np.repeat(unique_cards, n)
        cards = np.array_split(cards, len(unique_cards))
        cards = np.array(cards).tolist()
        return cards
    
    # Get all possible actions from the deck.
    def get_all_actions(self):
        unique_cards = np.arange(3, self.max_cards)
        single_cards = [[card] for card in unique_cards]
        double_cards = self.get_cards(2)
        triple_cards = self.get_cards(3)
        quad_cards = self.get_cards(4)
        actions = [[0], [2]] + single_cards + double_cards + \
                  triple_cards + quad_cards
        return actions
    
    # Get all possible playable actions from the hand.
    def get_playables(self):
        playables, counts = np.unique(self.hand, return_counts=True)
        actions = [[0]] # By default, all players can choose to pass.
        for playable, count in zip(playables, counts):
            if playable == 2:
                actions.append([playable])
            elif count > 1:
                for c in range(1, count + 1):
                    actions.append(list(np.repeat([playable], c)))
            else:
                actions.append([playable])
        return actions
    
    # Get random policy from a given hand and active card.
    def get_random_policy(self, active_card):
        actions = self.get_all_actions()
        policy = np.zeros(len(actions))
        indexes = [actions.index(action) for action 
                   in self.get_legal_actions(active_card)]
        policy[indexes] = 1/len(indexes)
        return policy
    
    # Get random policy from a completion event, given the active card.
    def get_random_completion_policy(self, active_card, count):
        actions = self.get_all_actions()
        policy = np.zeros(len(actions))
        playable = list(np.repeat(active_card[0], count))
        indexes = [actions.index(action) for action
                   in [[0], playable]]
        policy[indexes] = 1/2
        return policy
    
    # Play action given the policy and active card. 
    def play_action(self, policy, active_card):
        actions = self.get_all_actions() # Get all actions.
        action = np.random.choice(np.array(actions, dtype=object), p=policy)
        if action != [0]:
            # Remove cards from hand after playing the action.
            self.play_cards(action)
            # If the player bombs, the value of the active card is 0.
            active_card = [0] if action == [2] or len(action) == 4 else action
        return action, active_card

    # Play action from policy on a completion event, given the active card.
    def play_completion_action(self, policy, active_card):
        return self.play_action(policy, active_card)
