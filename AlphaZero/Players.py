from .MonteCarlo import MCTS
import numpy as np
import random


class AlphaZeroPlayer:
    def __init__(self, model):
        self.name = "AlphaZeroPlayer"
        self.model = model

    def get_action(self, game):
        # Create an instance of the MCTS class
        mcts = MCTS(game, self.model, self.model.args)

        # Get the probabilities of all possible actions
        action_probs = mcts.search()

        # Select the action with the highest probability
        best_move = np.argmax(action_probs)

        return best_move


class RandomPlayer:
    def __init__(self):
        self.name = "RandomPlayer"

    @staticmethod
    def get_action(game):
        if game.name == "G":
            legal_moves = game.get_legal_actions()
        else:
            legal_moves1, legal_moves2 = game.get_legal_actions()
            legal_moves = legal_moves1 + legal_moves2

        if legal_moves:
            move = random.choice(legal_moves)
            if game.name == "G":
                return game.encode(move)
            else:
                return game.encode(move[0], move[1])


class GreedyPlayer:
    def __init__(self):
        self.name = "GreedyPlayer"

    @staticmethod
    def get_action(game):
        if game.name == "G":
            legal_moves = game.get_legal_actions()
        else:
            legal_moves1, legal_moves2 = game.get_legal_actions()
            legal_moves = legal_moves1 + legal_moves2

        # Evaluate each move and select the one with the highest immediate reward
        best_move = None
        best_reward = float('-inf')

        for move in legal_moves:
            game_copy = game.clone()  # Ensure the original game state is not modified
            if game.name == "G":
                game_copy.apply_action(game_copy.encode(move))
            else:
                game_copy.apply_action(game_copy.encode(move[0], move[1]))
            reward = game_copy.get_score()

            if reward > best_reward:
                best_reward = reward
                best_move = move

        if game.name == "G":
            return game.encode(best_move)
        else:
            return game.encode(best_move[0], best_move[1])
