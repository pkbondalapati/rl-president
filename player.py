import numpy as np


class Player:
    def __init__(self, name, hand=[], rank='Person'):
        self.name = name
        self.hand = hand
        self.rank = rank
        self.score = 0
    
    # Get all active cards, given 'n' card count.
    def get_cards(self, n):
        unique_cards = np.arange(3, 15)
        cards = np.repeat(unique_cards, n)
        cards = np.array_split(cards, len(unique_cards))
        cards = np.array(cards).tolist()
        return cards
    
    # Get all possible actions from the deck.
    def get_all_actions(self):
        unique_cards = np.arange(3, 15)
        single_cards = [[card] for card in unique_cards]
        double_cards = self.get_cards(2)
        triple_cards = self.get_cards(3)
        quad_cards = self.get_cards(4)
        actions = [[0], [2]] + single_cards + double_cards + \
                  triple_cards + quad_cards
        return actions    
    
    # Get all playable actions from the player's hand.
    def get_playables(self):
        playables, counts = np.unique(self.hand, return_counts=True)
        actions = [[0]] # By default, all players can choose to pass.
        for playable, count in zip(playables, counts):
            if playable == 2:
                actions.append([playable])
            elif count > 1:
                actions.append(list(np.repeat([playable], count)))
            else:
                actions.append([playable])
        return actions
    
    # Get all legal actions from the player's hand. 
    def get_legal_actions(self, active_card):
        actions = self.get_playables()
        legal_actions = []
        for cards in actions:
            # Passing and playing 2s are legal actions.
            if cards[0] in [0, 2]:
                legal_actions.append(cards)
            
            # If the active card is singular, actions involving singular
            # cards that match or exceed the active card are legal. 
            elif len(active_card) == 1 and len(cards) == 1:
                if cards[0] >= active_card[0]:
                    legal_actions.append(cards)
            
            # If the active card is singular, actions involving multiple
            # cards are legal.
            elif len(active_card) == 1:
                legal_actions.append(cards)
            
            # If the active cards involve multiple cards...
            elif len(active_card) > 1:
                
                # If the action and active cards have the same card count, 
                # actions involving cards that match or exceed the active.
                # cards are legal.
                if len(cards) == len(active_card):
                    if cards[0] >= active_card[0]:
                        legal_actions.append(cards)
                
                # If the card count of the action exceeds that of the active
                # cards, then these actions are legal.
                elif len(cards) > len(active_card):
                    legal_actions.append(cards)
                
                # If the card count of the active cards is three, then actions
                # involving the remaining card that are legal.
                elif len(cards) == 1 and len(active_card) == 3:
                    if cards[0] == active_card[0]:
                        legal_actions.append(cards)
        return legal_actions
    
    # Reorder actions such that the first index contains pseudo-optimal action.
    def reorder_actions(self, actions):
        # Given that the actions are already sorted by numerical value, 
        # reorder actions by length of card count.
        actions.sort(key=lambda x:len(x))
        
        # If the player has only one legal action, return the actions.
        if len(actions) == 1:
            return actions
        
        # If the player is eligible to play a bomb and the player has more
        # than three legal actions, push the actions of playing a bomb and 
        # passing to the back.
        elif actions[1] == [2] and len(actions) > 3:
            actions = actions[2:len(actions)] + [[2]] + [[0]]
            
        # Otherwise, only push the action of passing to the back.
        else:
            actions = actions[1:len(actions)] + [[0]]
        return actions
    
    # Play pseudo-optimal action given the active card. 
    def play_action(self, active_card, random=False):
        actions = self.get_legal_actions(active_card) # Get all actions.
        if random:
            index = np.random.choice(range(len(actions)))
            action = actions[index]
        else:
            actions = self.reorder_actions(actions) # Reorder actions.
            action = actions[0] # Initial index is the optimal action.
        if action != [0]:
            # Remove cards from hand after playing the action.
            self.play_cards(action)
            # If the player bombs, the value of the active card is 0.
            active_card = [0] if action == [2] or len(action) == 4 else action
        return action, active_card
       
    # Play pseudo-optimal action on completion event.
    def play_completion(self, active_card, count):
        action = list(np.repeat(active_card[0], count))
        hand = np.array(self.hand)
        trim = hand[hand != 2]
        trim = trim[trim != active_card[0]]
        if len(trim) > 0 or 2 not in hand:
            self.play_cards(action)
            active_card = [0]
        return action, active_card
                
    # Remove cards from hand after playing the action.
    def play_cards(self, action):
        self.hand = np.array(self.hand)
        indexes = np.where(self.hand == action[0])[0]
        indexes = indexes[0:len(action)]
        self.hand = list(np.delete(self.hand, indexes))
    
    def __str__(self):
        return f"{self.name}: {list(self.hand)}"
    
    def __repr__(self):
        return self.__str__()
