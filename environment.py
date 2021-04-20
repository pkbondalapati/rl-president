import numpy as np


class Environment:
    def __init__(self, max_cards):
        self.max_cards = max_cards
        self.players = []
        self.history = []
    
    # Add player to environment.
    def add_player(self, name):
        player = Player(name, [])
        self.players.append(player)
    
    # Add multiple players to environment.
    def add_players(self, names):
        for name in names:
            self.add_player(name)
    
    # Deal out cards to all players in the environment.
    def deal_cards(self):
        suit = np.arange(2, self.max_cards)
        deck = np.repeat(suit, 4 * max(len(self.players)//4, 1))
        np.random.shuffle(deck)
        hands = np.array_split(deck, len(self.players))
        for player, hand in zip(self.players, hands):
            player.hand = sorted(hand)
    
    # Reorder players such that the one with the three of clubs is first.
    def reorder_players(self):
        count = []
        for player in self.players:
            hand = np.array(player.hand)
            count.append(len(hand[hand == 3]))
        sets = 4 * max(len(self.players)//4, 1)
        probs = np.array(count)/sets
        choice = np.random.choice(len(self.players), 1, p=probs)[0]
        self.players = self.players[choice:] + self.players[:choice]
    
    # Initialize the card game.
    def start_game(self):
        self.deal_cards()
        self.reorder_players()
        player = self.players[0]
        hand = np.array(player.hand)
        indexes = np.where(hand == 3)
        active_card = list(hand[indexes])
        player.hand = list(np.delete(hand, indexes))
        return active_card
    
    # Returns the count of duplicate cards from the last three plays.
    def get_prior_count(self):
        if len(self.history) < 1:
            return 0
        
        iter_ = len(self.history) - 1
        active_card = self.history[iter_]['Active Cards']
        counter = len(active_card)
        
        if len(self.history) > 1:
            prev_card = self.history[iter_ - 1]['Active Cards']

            if active_card[0] == prev_card[0]:
                counter += len(prev_card)

                if len(self.history) > 2:
                    prior_card = self.history[iter_ - 2]['Active Cards']

                    if prev_card[0] == prior_card[0]:
                        counter += len(prior_card)
                
        return counter
    
    # Find player that has a valid completion.
    def play_completions(self):
        counter = self.get_prior_count() 
        if counter != 0:
            iter_ = len(self.history) - 1
            active_card = self.history[iter_]['Active Cards']
            for player in self.players:
                hand = list(player.hand)
                cards, counts = np.unique(hand, return_counts=True)
                for card, count in zip(cards, counts):
                    if card == active_card[0] and count + counter == 4:                       
                        return player
    
    # Append play to the overall record of the card game.
    def update_history(self, turn, player, active_card):
        record = {'Turn': turn,
                  'Count': len(self.players),
                  'Name': player.name,
                  'Hand': player.hand,
                  'Active Cards': active_card}
        self.history.append(record)
    
    # Generate an episode of the game.
    def play(self, max_iter=100):
        active_card = self.start_game()
        player = self.players[0]
        turn = 1
        self.update_history(turn, player, active_card)
        
        while len(self.players) > 1:
                        
            iter_ = len(self.history) - 1
            turn = turn % len(self.players)
            
            player = self.players[turn]
            prev_card = self.history[iter_]['Active Cards']
            active_card = player.play_action(active_card)
            
            turn = turn + 1
            self.update_history(turn, player, active_card)
            
            if active_card == [0]:
                turn -= 1
            elif prev_card == active_card:
                turn += 1
            
            if iter_ > max_iter:
                break
                
    def __str__(self):
        return f"{self.players}"
    
    def __repr__(self):
        return self.__str__()
