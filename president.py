import numpy as np
from player import Player
from tf_agents.environments import py_environment
from tf_agents.environments import utils
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts


class PresidentEnv(py_environment.PyEnvironment):
    def __init__(self, policy):
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=49, name='action')
        # States 0-13: card counts, State 14: target value, 
        # State 15: target count, State 16: active players.
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(16,), dtype=np.int32, 
            minimum=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1], 
            maximum=[4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 14, 3, 4],
            name='state')
        self._episode_ended = False
        self.agent = None  # Tracks agent in the environment.
        self.players = []  # Tracks players in the environment.
        self.history = []  # Tracks cards played during episode.
        self.policy = True if policy == 'random' else False
        self.initialize()
    
    ## TF Agent Environment Methods
    
    def action_spec(self):
        return self._action_spec
    
    def observation_spec(self):
        return self._observation_spec
    
    def _reset(self):
        self._episode_ended = False
        self.agent = None
        self.players = []
        self.history = []
        self.initialize()
        
        # State at the start of the game.
        return ts.restart(np.array(self._state, dtype=np.int32))
    
    def _step(self, action):
                
        if self._episode_ended:
            # The last action ended the episode. Ignore the current action
            # and start a new episode.
            return self._reset()
        
        if not self.inactivated():
            turn = self.next_turn()
            target = self.get_target()

            # Check if legal move is played.
            if not self.check(action):
                self._episode_ended = True
                state = np.array(self._state, dtype=np.int32)
                return ts.termination(state, reward=(-10.0))
            else:
                action, target = self.agent.play_action(target)
                self.update_history(turn, self.agent, action, target)
                self.clear_pile()
                self.score_players()
                self.continue_play()
        
        state = np.array(self._state, dtype=np.int32)
        
        # Agent finishes game.
        if self._episode_ended:
            if len(self.agent.hand) == 0:
                # Ending on a bomb is a failure.
                if action == [2]:
                    return ts.termination(state, reward=(-1.0))
                # Agent finished first.
                elif len(self.players) == 3:
                    return ts.termination(state, reward=5.0)
                # Agent finished second.
                elif len(self.players) == 2: 
                    return ts.termination(state, reward=3.0)
                # Agent finished second to last.
                elif len(self.players) == 1: 
                    return ts.termination(state, reward=1.0)
            # Agent finished last.
            else:
                return ts.termination(state, reward=(-1.0))
        else:
            return ts.transition(state, reward=0.0, discount=1.0)

    ## Junction Methods
    
    # Starts game and updates history.
    def initialize(self):
        self.add_players(['Agent', 'Bella', 'Claire', 'David'])
        target = self.start_game()
        player = self.players[0]
        self.actions = player.get_all_actions()
        self.update_history(1, player, target, target)
        self.continue_play()
    
    # Plays cards until it is the agent's turn.
    def continue_play(self):        
        while len(self.players) > 1:
            self.play_completion()
            if self.inactivated():
                return
            turn = self.next_turn()
            player = self.players[turn - 1]
            target = self.history[-1]['Target']
            if player.name == 'Agent':
                state = self.vectorize(player.hand, target)
                self._state = np.array(state, dtype=np.int32)
                return
            action, target = player.play_action(target, random=self.policy)
            self.update_history(turn, player, action, target)
            self.clear_pile()
            self.score_players()
            if self.inactivated():
                return
        self._episode_ended = True
    
    # Check if action is legal; if not terminate episode.
    def check(self, index):
        target = self.get_target()
        playables = self.agent.get_legal_actions(target)
        action = self.actions[index]
        if action not in playables:
            return False
        return True
    
    # Check if agent is still active.
    def inactivated(self):
        agents = [p for p in self.players if p.name == 'Agent']
        if len(agents) == 0:
            self._episode_ended = True
            return True
        return False
    
    # Returns target from current state.
    def get_target(self):
        target = list(self._state[-3:-1])
        target = list(np.repeat(target[0], target[1]))
        return target
    
    # Track players that have finished the card game.
    def score_players(self):
        turn = self.history[-1]['Turn']
        hand = self.history[-1]['Hand']
        player = self.players[turn - 1]
        if len(hand) == 0:
            if player.name == 'Agent':
                self._episode_ended = True
            self.agent = player
            self.players.remove(player)
    
    # Returns a list of card counts for values 2 to 14.
    def vectorize(self, hand, target):
        hand_vec = self.hand2vec(hand)
        card_vec = self.card2vec(target)
        count_vec = [len(self.players)]
        vector = hand_vec + card_vec + count_vec
        return vector
    
    ## Python Environment Methods
    
    # Add players to environment.
    def add_players(self, names):
        for name in names:
            player = Player(name)
            if name == 'Agent':
                self.agent = player
                self.players.append(self.agent)
            else:
                self.players.append(player)
    
    # Deal out cards to all players in the environment.
    def deal_cards(self):
        suit = np.arange(2, 15)
        deck = np.repeat(suit, 4)
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
        probs = np.array(count)/4
        choice = np.random.choice(len(self.players), 1, p=probs)[0]
        self.players = self.players[choice:] + self.players[:choice]
    
    # Initialize the card game.
    def start_game(self):
        self.deal_cards()
        self.reorder_players()
        player = self.players[0]
        hand = np.array(player.hand)
        indexes = np.where(hand == 3)[0]
        action = list(np.repeat(3, indexes.shape[0]))
        player.play_cards(action)
        target = [0] if len(action) == 4 else action
        return target
    
    # Returns the running card count of the past 'n' plays.
    def previous_count(self, iter_, n, pass_):
        length = len(self.history) - iter_
        if iter_ < 0 or length > n:
            return 0

        target = self.history[iter_]['Target']
        nonzero_action = target[0] != 0

        if iter_ > 0:
            prev_card = self.history[iter_ - 1]['Target']
            match_cards = target[0] == prev_card[0]
            pass_action = len(target) == len(prev_card) \
                if pass_ else True

            if match_cards and nonzero_action and pass_action:
                basic_count = 1 if pass_ else len(target)
                return basic_count + self.previous_count(iter_ - 1, n, pass_)

        default_count = 0 if pass_ else len(target)
        return default_count
    
    # Find player that has a valid completion and play that completion.
    def play_completion(self):
        iter_ = len(self.history) - 1
        counter = self.previous_count(iter_, 3, False)
        target = self.history[iter_]['Target']
        for index, player in enumerate(self.players):
            hand = list(player.hand)
            cards, counts = np.unique(hand, return_counts=True)
            for card, count in zip(cards, counts):
                if card == target[0] and count + counter == 4:
                    action, target = player.play_completion(target, count)
                    if target == [0]:
                        index = self.players.index(player)
                        self.update_history(index + 1, player, action, target)
                        self.score_players()
            
    # Get next turn for the card game.
    def next_turn(self):
        iter_ = len(self.history) - 1
        hand = self.history[iter_]['Hand']
        action = self.history[iter_]['Action']
        target = self.history[iter_]['Target']
        turn = self.history[iter_]['Turn']
        turn_ = (turn % len(self.players)) + 1
        
        # Check if player finished.
        if len(hand) == 0:
            turn = max((turn % (len(self.players) + 1)), 1)
        
        # Check if player skips.
        elif self.previous_count(iter_, 1, True) == 1 and action != [0]:
            turn = (turn_ % len(self.players)) + 1
            
        # Check if player neither bombs nor completes or just passes.
        elif (action != [2] and target != [0]) or action == [0]:
            turn = turn_
            
        # Otherwise, return same turn.
        return turn
    
    # Reset target to [0] if all players pass. 
    def clear_pile(self):
        count = len(self.players)
        if len(self.history) >= count:
            iter_ = len(self.history) - 1
            prev_actions = [self.history[i]['Action'] 
                            for i in range(iter_, iter_ - count, -1)]
            if all(action == [0] for action in prev_actions):
                self.history[iter_]['Target'] = [0]
    
    # Append play to the overall record of the card game.
    def update_history(self, turn, player, action, target):
        record = {'Turn': turn,
                  'Count': len(self.players),
                  'Name': player.name,
                  'Hand': player.hand,
                  'Action': action,
                  'Target': target}
        self.history.append(record)
        
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

if __name__ == '__main__':
    env = PresidentEnv(policy='optimal')
    utils.validate_py_environment(env, episodes=1000)
    print(env.history)
