import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self, state_dim, num_layers, hidden_dim):
        super(CNN, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_layers)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = F.softmax(x, dim=-1)
        return x


class MetaLearner():
    def __init__(self, state_dim, hidden_dim, num_layers, device):
        self.device = device
        self.input_dim = state_dim  # state, action, and failure signal
        self.hidden_dim = hidden_dim

        self.cnn = CNN(state_dim, num_layers, hidden_dim).to(self.device)

        self.optimizer = optim.Adam(self.cnn.parameters(), lr=0.001)
        self.loss_fn = nn.MSELoss()

    def update(self, states, reward):
        # Combine states, actions, and failure signals into a single tensor

        self.optimizer.zero_grad()

        rew_r = self.cnn(states)
        reward_clone = reward.clone().detach().requires_grad_(True)
        rew = torch.sum(rew_r * reward_clone)

        mse_loss = nn.MSELoss()
        loss = mse_loss(rew/50, torch.tensor(1.).expand_as(rew).to(self.device))

        loss.backward()
        self.optimizer.step()

        return rew_r, loss

    def act(self, x):
        return self.cnn(x)
