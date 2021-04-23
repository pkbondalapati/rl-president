import numpy as np
from player import Player


class Environment:
    def __init__(self, max_cards):
        self.max_cards = max_cards
        self.players = []
        self.history = []
        self.results = []
        self.ranks = []
        self.fails = []
    
    # Add player to environment.
    def add_player(self, name, rank='Person'):
        player = Player(name, [], rank)
        self.players.append(player)
    
    # Add multiple players to environment.
    def add_players(self, names, ranks):
        for name, rank in zip(names, ranks):
            self.add_player(name, rank)
    
    # Deal out cards to all players in the environment.
    def deal_cards(self):
        suit = np.arange(2, self.max_cards)
        deck = np.repeat(suit, 4 * max(len(self.players)//4, 1))
        np.random.shuffle(deck)
        hands = np.array_split(deck, len(self.players))
        for player, hand in zip(self.players, hands):
            player.hand = sorted(hand)
    
    # Swap 'n' cards in relation to rank.
    def trade(self, primary, secondary, n):
        primary_hand = np.array(primary.hand)
        primary_bombs = primary_hand[primary_hand == 2]
        primary_hand = primary_hand[primary_hand != 2]
        primary_trade = list(primary_hand[:n])
        primary_hand = list(primary_bombs) + list(primary_hand[n:])
        secondary_hand = np.array(secondary.hand)
        secondary_bombs = secondary_hand[secondary_hand == 2]
        secondary_hand = secondary_hand[secondary_hand != 2]
        secondary_trade = list(secondary_hand[-n:])
        secondary_hand = list(secondary_bombs) + list(secondary_hand[:-n])
        primary.hand = sorted(primary_hand + secondary_trade)
        secondary.hand = sorted(secondary_hand + primary_trade)
    
    # Trade cards according to the players' rank.
    def trade_cards(self):
        president, vice_president, vice_scrub, scrub = [None] * 4
        for player in self.players:
            # Get titles from players.
            if player.rank == "President":
                president = player
            elif player.rank == "Vice President":
                vice_president = player
            elif player.rank == "Vice Scrub":
                vice_scrub = player
            elif player.rank == "Scrub":
                scrub = player
        
        # Trade cards between president and scrub.
        if all(player is not None 
               for player in [president, scrub]):
            n = 1 if len(self.players) < 4 else 2
            self.trade(president, scrub, n)
        elif all(player is not None 
                 for player in [vice_president, vice_scrub]):
            self.trade(vice_president, vice_scrub, n=1)
    
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
        self.trade_cards()
        self.reorder_players()
        player = self.players[0]
        hand = np.array(player.hand)
        indexes = np.where(hand == 3)[0]
        action = list(np.repeat(3, indexes.shape[0]))
        player.play_cards(action)
        return action
    
    # Returns the running card count of the past 'n' plays.
    def previous_count(self, iter_, n, pass_):
        length = len(self.history) - iter_
        if iter_ < 0 or length > n:
            return 0

        active_card = self.history[iter_]['Active Cards']
        nonzero_action = active_card[0] != 0

        if iter_ > 0:
            prev_card = self.history[iter_ - 1]['Active Cards']
            match_cards = active_card[0] == prev_card[0]
            pass_action = len(active_card) == len(prev_card) if pass_ else True

            if match_cards and nonzero_action and pass_action:
                basic_count = 1 if pass_ else len(active_card)
                return basic_count + self.previous_count(iter_ - 1, n, pass_)

        default_count = 0 if pass_ else len(active_card)
        return default_count
    
    # Find player that has a valid completion and play that completion.
    def play_completion(self):
        iter_ = len(self.history) - 1
        counter = self.previous_count(iter_, 3, False)
        active_card = self.history[iter_]['Active Cards']
        for index, player in enumerate(self.players):
            hand = list(player.hand)
            cards, counts = np.unique(hand, return_counts=True)
            for card, count in zip(cards, counts):
                if card == active_card[0] and count + counter == 4:
                    turn = index + 1
                    action = list(np.repeat(card, count))
                    player.play_cards(action)
                    active_card = [0]
                    self.update_history(turn, player, action, active_card)
            
    # Get next turn for the card game.
    def get_next_turn(self):
        iter_ = len(self.history) - 1
        hand = self.history[iter_]['Hand']
        action = self.history[iter_]['Action']
        active_card = self.history[iter_]['Active Cards']
        turn = self.history[iter_]['Turn']
        next_turn = (turn % len(self.players)) + 1
        
        # Check if player skips.
        if self.previous_count(iter_, 1, True) == 1:
            turn = (next_turn % len(self.players)) + 1
            
        # Check if player does neither bombs nor completes or just passes.
        elif (action != [2] and active_card != [0]) or \
             action == [0] or len(hand) == 0:
            turn = next_turn
        
        # Otherwise, return same turn.
        return turn
    
    # Reset active card to [0] if all players pass. 
    def clear_pile(self):
        count = len(self.players)
        if len(self.history) >= count:
            iter_ = len(self.history) - 1
            prev_actions = [self.history[i]['Action'] 
                            for i in range(iter_, iter_ - count, -1)]
            if all(action == [0] for action in prev_actions):
                self.history[iter_]['Active Cards'] = [0]
    
    # Track players that have finished the card game.
    def score_players(self):
        iter_ = len(self.history) - 1
        turn = self.history[iter_]['Turn']
        hand = self.history[iter_]['Hand']
        action = self.history[iter_]['Action']
        player = self.players[turn - 1]
        if len(hand) == 0:
            if action == [2]:
                self.fails.append(player)
            else:
                self.ranks.append(player)
            
            # Remove player from the environment.
            self.players.remove(player)
    
    # Rank all players after the card game.
    def update_results(self):
        players = self.ranks + self.players + self.fails
        president = players[0]
        president.rank = 'President'
        president.score += 2
        scrub = players[-1]
        scrub.rank = 'Scrub'
        
        if len(players) > 3:
            vice_president = players[1]
            vice_president.rank = 'Vice President'
            vice_president.score += 1
            vice_scrub = players[-2]
            vice_scrub.rank = 'Vice Scrub'
            for i, player in enumerate(players[2:-2]):
                player.rank = f'Person {i + 1}'
        else:
            for i, player in enumerate(players[1:-1]):
                player.rank = f'Person {i + 1}'
        
        self.results = {player.rank: player.name 
                        for player in players}
        players.sort(key=lambda x: x.name)
        self.players = players
    
    # Append play to the overall record of the card game.
    def update_history(self, turn, player, action, active_card):
        record = {'Turn': turn,
                  'Count': len(self.players),
                  'Name': player.name,
                  'Hand': player.hand,
                  'Action': action,
                  'Active Cards': active_card}
        self.history.append(record)
        
    # Run an iteration of a single game.
    def play(self, max_iter=100):
        action = self.start_game()
        player = self.players[0]
        active_card = action 
        self.update_history(1, player, action, active_card)
        
        while len(self.players) > 1:
            
            self.play_completion()
            iter_ = len(self.history) - 1
            turn = self.get_next_turn()
            player = self.players[turn - 1]
            active_card = self.history[iter_]['Active Cards']
            action, active_card = player.play_action(active_card)
            self.update_history(turn, player, action, active_card)
            self.clear_pile()
            self.score_players()
            
            if iter_ > max_iter:
                break
        
        self.update_results()
    
    # Generate an episode of a card game session.
    def get_episode(self, threshold):
        scores = [player.score for player in self.players]
        result_record = []
        score_record = []
        history_record = []
        
        while max(scores) < threshold:
            self.play()
            result_record.append(self.results)
            history_record.append(self.history)
            scores = [player.score for player in self.players]
            score_record.append(scores)
            self.history = []
            self.ranks = []
            self.fails = []
        
        return (result_record, score_record, history_record)
                
    def __str__(self):
        return f"{self.results}"
    
    def __repr__(self):
        return self.__str__()
