from .Players import RandomPlayer, GreedyPlayer, AlphaZeroPlayer
from multiprocessing import Pool
from .MonteCarlo import MCTS
from .CNNET import CNNET
from tqdm import tqdm
import numpy as np
import logging
import torch
import csv
import os


class AlphaZero:
    def __init__(self, game, args):
        self.game = game
        self.args = args
        self.model_name = f"{game.name}{game.n}x{game.m}"
        self.win_rates = {'RandomPlayer': 0.0, 'GreedyPlayer': 0.0, 'AlphaZeroPlayer': 0.0}
        self.cnnet = CNNET(game, args).to(args['device'])
        self.mcts = MCTS(game, self.cnnet, self.args)
        self.iter = None

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def self_play(self):
        # Create a copy of the game to play out the self-play episode
        self_play_game = self.game.clone()

        # Create a new MCTS instance for self-play
        self_play_mcts = MCTS(self_play_game, self.cnnet, self.args)

        training_data = []

        while not self_play_game.is_terminal():
            action_prob = self_play_mcts.search()
            training_data.append(
                (self_play_game.board * self_play_game.player_turn, action_prob, self_play_game.player_turn))
            # print(np.array(self_play_game.board))
            # print()

            # Apply the selected action to the state
            action = np.random.choice(self_play_game.getActionSize(), p=action_prob)
            self_play_game.apply_action(action)

        # Collect the training data for the self-play episode
        adjusted_training_data = []
        for hist_neutral_state, hist_action_probs, hist_player in training_data:
            hist_outcome = self_play_game.winner if hist_player == self.game.player_turn else -self_play_game.winner
            augmented_states = [hist_neutral_state]

            for augmented_state in augmented_states:
                augmented_state = np.array(augmented_state)
                if len(augmented_state) == 0:
                    continue
                adjusted_training_data.append(
                    (self.game.get_encoded_board(augmented_state), hist_action_probs, hist_outcome))

        return adjusted_training_data

    def learn(self):
        """
        Learning loop for AlphaZero
        """
        if self.iter is None:
            self.iter = 0

        if self.iter >= self.args["iterations_limit"]:
            print(f"Limit {self.args['iterations_limit']} iteractions reached!!!!")
            return

        with Pool(self.args["num_processes"]) as pool:
            try:
                while self.iter < self.args["iterations_limit"]:
                    self.iter += self.args['num_episodes']
                    episodes_bar = tqdm(total=self.args["num_episodes"], desc=f'Iteration {self.iter}: Episodes',
                                        position=0, leave=True)
                    for episode in range(self.args["num_episodes"]):
                        training_data = pool.apply_async(self.self_play).get()
                        self.cnnet.train_model(training_data)
                        episodes_bar.update(1)

                    self.evaluate_and_save_model()
                    episodes_bar.close()

                print(f"Limit {self.args['iterations_limit']} iteractions reached!!!!")
            except Exception as e:
                print(f"An error occurred: {e}")

    def validation_games(self, player1, player2, model_turn):
        total_wins = 0
        validation_game = self.game.clone()
        validation_game.play_game(player1, player2)
        if validation_game.winner == model_turn:
            total_wins += 1
            return total_wins
        else:
            return total_wins

    def evaluate_and_save_model(self):
        total_wins_random = 0
        total_wins_greedy = 0
        total_wins_alphazero = 0

        alphazero_trained = AlphaZeroPlayer(self.cnnet)

        random_player = RandomPlayer()
        greedy_player = GreedyPlayer()
        alphazero_baseline = AlphaZeroPlayer(CNNET(self.game, self.args).to(self.args['device']))

        for i in range(self.args["validation_episodes"]):
            if i % 2 == 0:
                total_wins_random += self.validation_games(alphazero_trained, random_player, 1)
                total_wins_greedy += self.validation_games(alphazero_trained, greedy_player, 1)
                total_wins_alphazero += self.validation_games(alphazero_trained, alphazero_baseline, 1)
            else:
                total_wins_random += self.validation_games(random_player, alphazero_trained, -1)
                total_wins_greedy += self.validation_games(greedy_player, alphazero_trained, -1)
                total_wins_alphazero += self.validation_games(alphazero_baseline, alphazero_trained, -1)

        self.win_rates['RandomPlayer'] = total_wins_random / self.args["validation_episodes"]
        self.win_rates['GreedyPlayer'] = total_wins_greedy / self.args["validation_episodes"]
        self.win_rates['AlphaZeroPlayer'] = total_wins_alphazero / self.args["validation_episodes"]

        if self.args["training_verbose"]:
            self.logger.info(f"Validation Win Rate against RandomPlayer: {self.win_rates['RandomPlayer'] * 100:.2f}%")
            self.logger.info(f"Validation Win Rate against GreedyPlayer: {self.win_rates['GreedyPlayer'] * 100:.2f}%")
            self.logger.info(
                f"Validation Win Rate against AlphaZero baseline: {self.win_rates['AlphaZeroPlayer'] * 100:.2f}%")

        self.save_model()

    def save_model(self):
        # print("Saving model...")
        save_dir = f"{self.args['best_model_dir']}{self.model_name}/"
        os.makedirs(save_dir, exist_ok=True)
        torch.save(self.cnnet.state_dict(), f"{save_dir}model.tar")
        self.save_best_win_rates()

    def save_best_win_rates(self):
        # Save the best win rates to a CSV file
        csv_file = f"{self.args['best_model_dir']}{self.model_name}/win_rates.csv"
        with open(csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            # If it's the first iteration, write the header
            if self.iter == self.args['num_episodes']:
                writer.writerow(['n_iter', 'RandomPlayer', 'GreedyPlayer', 'AlphaZeroBaseline'])
            # Write the best win rates
            writer.writerow([self.iter, self.win_rates['RandomPlayer'], self.win_rates['GreedyPlayer'],
                             self.win_rates['AlphaZeroPlayer']])

    def load_model(self):
        models_dir = f"{self.args['best_model_dir']}{self.model_name}/"
        os.makedirs(models_dir, exist_ok=True)
        models = os.listdir(models_dir)
        if len(models) == 0:
            return False
        self.iter = self.get_last_iter(f"{models_dir}win_rates.csv")
        model_path = f"{models_dir}model.tar"
        return self.cnnet.load_model(model_path)

    @staticmethod
    def get_last_iter(csv_file):
        with open(csv_file, 'r') as file:
            reader = csv.reader(file)
            rows = list(reader)
            last_row = rows[-1]
            n_iter = int(last_row[0])
            return n_iter
