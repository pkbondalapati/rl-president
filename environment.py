import numpy as np
from player import Player
from agent import Agent


class Environment:
    def __init__(self, max_cards):
        self.max_cards = max_cards
        self.players = []
        self.history = []
        self.results = []
        self.vectors = []
        self.weights = []
        self.records = []
        self.rewards = []
        self.shadows = []
        self.ranks = []
        self.fails = []
        self.completion = False
    
    # Add player to environment.
    def add_player(self, name, rank='Person'):
        player = Agent(self.max_cards, rank=rank) if name == 'Agent' \
            else Player(name, rank)
        self.players.append(player)
        self.shadows.append(player)
    
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

        active_card = self.history[iter_]['Target']
        nonzero_action = active_card[0] != 0

        if iter_ > 0:
            prev_card = self.history[iter_ - 1]['Target']
            match_cards = active_card[0] == prev_card[0]
            pass_action = len(active_card) == len(prev_card) \
                if pass_ else True

            if match_cards and nonzero_action and pass_action:
                basic_count = 1 if pass_ else len(active_card)
                return basic_count + self.previous_count(iter_ - 1, n, pass_)

        default_count = 0 if pass_ else len(active_card)
        return default_count
    
    # Returns true if agent has a valid completion, otherwise false.
    def has_completion(self):
        iter_ = len(self.history) - 1
        counter = self.previous_count(iter_, 3, False)
        active_card = self.history[iter_]['Target']
        agents = [p for p in self.players if p.name == 'Agent']
        if len(agents) != 0:
            hand = list(agents[0].hand)
            cards, counts = np.unique(hand, return_counts=True)
            for card, count in zip(cards, counts):
                if card == active_card[0] and count + counter == 4:
                    return True
        return False
    
    # Find player that has a valid completion and play that completion.
    def play_completion(self):
        if not self.has_completion():
            iter_ = len(self.history) - 1
            counter = self.previous_count(iter_, 3, False)
            active_card = self.history[iter_]['Target']
            for index, player in enumerate(self.players):
                hand = list(player.hand)
                cards, counts = np.unique(hand, return_counts=True)
                for card, count in zip(cards, counts):
                    if card == active_card[0] and count + counter == 4:
                        action, active_card = \
                        player.play_completion(active_card, count)                        
                        if active_card == [0]:
                            index = self.players.index(player)
                            self.update_history(index + 1, player, 
                                                action, active_card)
                            self.score_players()
            
    # Get next turn for the card game.
    def get_next_turn(self):
        iter_ = len(self.history) - 1
        name = self.history[iter_]['Name']
        hand = self.history[iter_]['Hand']
        action = self.history[iter_]['Action']
        active_card = self.history[iter_]['Target']
        turn = self.history[iter_]['Turn']
        next_turn = (turn % len(self.players)) + 1
        
        # Check if player finished.
        if len(hand) == 0:
            turn = max((turn % (len(self.players) + 1)), 1)
        
        # Check if player skips.
        elif self.previous_count(iter_, 1, True) == 1 and action != [0]:
            turn = (next_turn % len(self.players)) + 1
            
        # Check if player neither bombs nor completes or just passes.
        elif (action != [2] and active_card != [0]) or action == [0]:
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
                self.history[iter_]['Target'] = [0]
    
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
            self.update_shadows()
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
                  'Target': active_card}
        self.history.append(record)
        self.update_shadows()
    
    # Append vector to the list of vectors.
    def update_vectors(self, player, active_card, action=[]):
        vector = self.vectorize(player.hand, active_card, action)
        reward = self.get_reward(vector)
        self.vectors.append(vector)
    
    # Updates the Player object's copy. 
    def update_shadows(self):
        for player in self.players:
            for shadow in self.shadows:
                if player.name == shadow.name:
                    shadow = player
    
    # Starts game and updates history.
    def initialize(self):
        action = self.start_game()
        player = self.players[0]
        active_card = action 
        self.update_history(1, player, action, active_card)
    
    # Run an iteration of a single game.
    def play(self, max_iter=100):
        self.initialize()
        
        while len(self.players) > 1:
            
            self.play_completion()
            iter_ = len(self.history) - 1
            turn = self.get_next_turn()
            player = self.players[turn - 1]
            active_card = self.history[iter_]['Target']
            if player.name == 'Agent':
                self.update_vectors(player, active_card)
                policy = player.get_random_policy(active_card)
                action, active_card = player.play_action(policy, active_card)
            else:
                action, active_card = player.play_action(active_card)
            self.update_history(turn, player, action, active_card)
            self.clear_pile()
            self.score_players()
            
            if iter_ > max_iter:
                break
        
        self.update_results()
    
    # Play a single step of in the episode.
    def play_step(self):
        if len(self.history) == 0:
            self.initialize()
        
        while len(self.players) > 1:
            iter_ = len(self.history) - 1
            active_card = self.history[iter_]['Target']
            
            if self.has_completion() and not self.completion:
                player = [p for p in self.players if p.name == 'Agent'][0]
                hand = np.array(player.hand)
                count = len(hand[hand == active_card[0]])
                
                actions = player.get_completion_actions(active_card, count)
                vectors = [self.vectorize(hand, active_card, action) 
                           for action in actions]
                values = [self.evaluate_Q(vector)
                          for vector in vectors]
                
                action = player.epsilon_greedy(0.2, actions, values)
                active_card = [0] if action != [0] else active_card
                self.completion = True if action == [0] else False
                
                state_action = self.vectorize(player.hand, [0], action)
                reward = self.get_reward(state_action)
                self.rewards.append(reward)
                self.records.append(state_action)
                
                if action != [0]:
                    player.play_cards(action)                    
                    index = self.players.index(player)
                    self.update_history(index + 1, player, 
                                        action, active_card)
                    self.score_players()
                break
            
            self.completion = False
            self.play_completion()
            turn = self.get_next_turn()
            player = self.players[turn - 1]
            active_card = self.history[iter_]['Target']
            if player.name == 'Agent':
                actions = player.get_legal_actions(active_card)
                vectors = [self.vectorize(player.hand, 
                                          active_card, action)
                           for action in actions]
                values = [self.evaluate_Q(vector)
                          for vector in vectors]
                action = player.epsilon_greedy(0.2, actions, values)
                player.play_cards(action)
                if action != [0]:
                    active_card = [0] if (action == [2] or 
                                          len(action) == 4) else action
                
                state_action = self.vectorize(player.hand, active_card, action)
                reward = self.get_reward(state_action)
                self.rewards.append(reward)
                self.records.append(state_action)

                self.update_history(turn, player, action, active_card)
                self.clear_pile()
                self.score_players()
                break
            else:
                action, active_card = player.play_action(active_card)
                self.update_history(turn, player, action, active_card)
                self.clear_pile()
                self.score_players()
    
    # Returns output from the action-value function.
    def evaluate_Q(self, vector):
        if len(self.weights) == 0:
            self.weights = np.zeros(21)
#             self.weights = np.ones(20)
#             self.weights = np.random.normal(0.01, 0.05, 20)
        assert len(vector) == len(self.weights), 'Mismatch of vector length.'
        return np.dot(np.array(self.weights).T, np.array(vector))
    
    # Runs SARSA for a specified number of episodes.
    def sarsa(self, alpha, gamma, max_episodes=1):
        for _ in range(max_episodes):
            # Initial State
            self.play_step()
            prev_vector = np.array(self.records[-1])
            
            counter = 0
            while counter < len(self.rewards):
                self.play_step()
                next_vector = np.array(self.records[-1])
                reward = self.rewards[-1]
                
                prev_value = self.evaluate_Q(prev_vector)
                next_value = self.evaluate_Q(next_vector)
                
                self.weights = self.weights + alpha * \
                    (reward + gamma * next_value - prev_value) * \
                    prev_vector
                
                prev_vector = next_vector
                counter += 1
            
            # Terminal State
            self.weights = self.weights + alpha * \
                (reward - prev_value) * prev_vector
            
            # Reset Game
            self.reset()
    
    def reset(self):
        self.players = self.shadows
        self.history = []
        self.results = []
        self.records = []
        self.rewards = []
        self.ranks = []
        self.fails = []
        self.completion = False
    
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
    
    # Convert hand to vector.
    def hand2vec(self, hand):
        vector = list(np.zeros(13, dtype=int))
        suite = list(range(2, 15))
        values, counts = np.unique(hand, return_counts=True)
        indexes = [(suite.index(v), c) for v, c in zip(values, counts)]
        for index, count in indexes:
            vector[index] = count
        return vector
    
    # Convert cards to vector.
    def card2vec(self, cards):
        if len(cards) == 0:
            return []
        values, counts = np.unique(cards, return_counts=True)
        assert len(values) == 1 and len(counts) == 1, 'Invalid card/action.'
        return [values[0], counts[0]]
    
    # Convert players' card counts to vector.
    def count2vec(self):
        return [len(p.hand) for p in self.shadows if p.name != 'Agent']
    
    # Returns a list of card counts for values 2 to 14.
    def vectorize(self, hand, active_card, action=[]):
        hand_vec = self.hand2vec(hand)
        card_vec = self.card2vec(active_card)
        count_vec = self.count2vec()
        action_vec = self.card2vec(action) if len(action) != 0 else action
        vector = [1] + hand_vec + card_vec + action_vec + count_vec
        return vector
    
    # Get reward from vector representation of (state, action).
    def get_reward(self, vector):
        card_count = sum(vector[0:13])
        players_count = vector[-len(self.shadows):]
        reward = 0
        if card_count == 0:
            reward += 1
        nonzeros = len(np.nonzero(players_count)[0])
        if card_count == 0 and nonzeros == 3:
            reward += 1
        elif nonzeros == 1:
            reward = -2
        return reward
        
    def __str__(self):
        return f"{self.results}"
    
    def __repr__(self):
        return self.__str__()
