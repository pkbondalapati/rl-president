import random
import numpy as np

def deal(num_players):
    '''Basic card game dealing'''


    suits = ['heart', 'diamonds', 'spades', 'clubs']
    deck = [(value, suit) for value in range(1, 14) for suit in suits]
    #1 = Ace, 11 = Jack, 12 = Queen, 13 = King

    random.shuffle(deck)

    hands = [[] for i in range(num_players)]
    remainder = len(deck) % num_players
    num_cards = len(deck) // num_players

    card_idx = 0
    for player in range(num_players):
        hands[player].extend(deck[card_idx:card_idx + num_cards])
        card_idx += num_cards

    for i in range(remainder):
        hands[i].extend(deck[card_idx:card_idx + 1])
        card_idx += 1

    return(hands)

    #https://stackoverflow.com/questions/41970795/what-is-the-best-way-to-create-a-deck-of-cards/41970851


def president_deal(num_players):
    '''Dealing specific to President game, since suit does not matter'''

    dealt = deal(num_players)

    pres_hands = [[] for i in range(len(dealt))]

    i = 0
    for hand in dealt:
        card_nums = [h[0] for h in hand]
        pres_hands[i] = [[x, card_nums.count(x)] for x in set(range(1, 14))]
        i += 1

    return(pres_hands)
