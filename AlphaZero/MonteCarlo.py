import numpy as np
import torch
import math


class Node:
    def __init__(self, game, args, parent=None, action_taken=None, prior=0, visit_count=0):
        self.game = game
        self.args = args
        self.parent = parent
        self.action_taken = action_taken
        self.prior = prior
        self.children = []
        self.visit_count = visit_count
        self.value_sum = 0

    def is_expanded(self):
        return len(self.children) > 0

    def select(self):
        best_child = None
        best_ucb = -np.inf

        for child in self.children:
            ucb = self.get_ucb(child)
            if ucb > best_ucb:
                best_child = child
                best_ucb = ucb

        return best_child

    def get_ucb(self, child):
        if child.visit_count == 0:
            q_value = 0
        else:
            q_value = 1 - ((child.value_sum / child.visit_count) + 1) / 2
        return q_value + self.args['UCB_exploration_weight'] * (math.sqrt(self.visit_count) / (child.visit_count + 1)) * child.prior

    def expand(self, policy):
        for action, prob in enumerate(policy):
            if prob > 0:
                child_game = self.game.clone()
                child_game.apply_action(action)

                child = Node(child_game, self.args, parent=self, action_taken=action, prior=prob)
                self.children.append(child)

    def backpropagate(self, value):
        self.value_sum += value
        self.visit_count += 1

        if self.parent is not None:
            self.parent.backpropagate(-value)


class MCTS:
    def __init__(self, game, model, args):
        self.model = model
        self.game = game
        self.args = args

    @torch.no_grad()
    def search(self):
        root = Node(self.game, self.args, visit_count=1)
        policy, _ = self.model(torch.tensor(self.game.get_encoded_board(), device=self.model.device).unsqueeze(0))
        policy = torch.softmax(policy, dim=1).squeeze(0).cpu().numpy()
        policy = (1 - self.args['dirichlet_epsilon']) * policy + self.args['dirichlet_epsilon'] * np.random.dirichlet([self.args['dirichlet_alpha']] * self.game.getActionSize())
        valid_moves = self.game.get_encoded_actions()
        policy *= valid_moves
        policy /= np.sum(policy)
        root.expand(policy)
        for search in range(self.args['num_mcts_sims']):
            node = root
            while node.is_expanded():
                node = node.select()
            is_terminal = node.game.game_over
            value = node.game.winner
            if not is_terminal:
                policy, value = self.model(torch.tensor(node.game.get_encoded_board(), device=self.model.device).unsqueeze(0))
                policy = torch.softmax(policy, dim=1).squeeze(0).cpu().numpy()
                valid_moves = node.game.get_encoded_actions()
                policy *= valid_moves
                policy /= np.sum(policy)
                value = value.item()
                node.expand(policy)
            node.backpropagate(value)
        action_probs = np.zeros(self.game.getActionSize())
        for child in root.children:
            action_probs[child.action_taken] = child.visit_count
        action_probs /= np.sum(action_probs)
        return action_probs
