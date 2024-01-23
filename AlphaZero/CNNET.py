import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import random
import torch
import os


class CNNET(nn.Module):
    def __init__(self, game, args):
        super(CNNET, self).__init__()
        self.game = game
        self.args = args
        self.n = game.n
        self.m = game.m
        self.device = self.args['device']
        # Define your convolutional layers
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)

        # Define fully connected layers
        self.fc1 = nn.Linear(256 * self.n * self.m, 512)
        self.fc2 = nn.Linear(512, game.getActionSize())

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.args['learning_rate'])

    def forward(self, x):
        # Input x should be the game state
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # Flatten the output for fully connected layers
        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        # Compute the policy and value
        policy = F.softmax(x, dim=1)
        value = x.sum(dim=1, keepdim=True)  # keep dimensions to match with value_targets

        return policy, value

    def train_model(self, training_data):
        random.shuffle(training_data)
        for batchIdx in range(0, len(training_data), self.args['batch_size']):
            sample = training_data[batchIdx:batchIdx + self.args['batch_size']]
            state, policy_targets, value_targets = zip(*sample)

            state, policy_targets, value_targets = np.array(state), np.array(policy_targets), np.array(value_targets).reshape(-1, 1)

            state = torch.tensor(state, dtype=torch.float32, device=self.device)
            policy_targets = torch.tensor(policy_targets, dtype=torch.float32, device=self.device)
            value_targets = torch.tensor(value_targets, dtype=torch.float32, device=self.device)

            out_policy, out_value = self(state)

            policy_loss = F.cross_entropy(out_policy, policy_targets)
            value_loss = F.mse_loss(out_value, value_targets)
            loss = policy_loss + value_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def load_model(self, model_path):
        if os.path.exists(model_path):
            self.load_state_dict(torch.load(str(model_path), map_location=self.args["device"]))
            return True
        else:
            return False
