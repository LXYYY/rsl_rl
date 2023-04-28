import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class LSTM(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_layers=2):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = x.unsqueeze(0).unsqueeze(0)
        # Initialize hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)

        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        out = F.softmax(out, dim=1)
        return out.squeeze(0)


class CNN(nn.Module):
    def __init__(self, state_dim, output_size, hidden_dim):
        super(CNN, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = F.softmax(x, dim=0)
        return x


class MetaLearner():
    def __init__(self, state_dim, hidden_dim, output_size, device, use_lstm=False):
        self.device = device
        self.input_dim = state_dim  # state, action, and failure signal
        self.hidden_dim = hidden_dim

        if not use_lstm:
            self.network = CNN(state_dim, output_size, hidden_dim).to(self.device)
        else:
            print("MetaLearner using LSTM")
            self.network = LSTM(state_dim, output_size, hidden_dim).to(self.device)

        self.optimizer = optim.Adam(self.network.parameters(), lr=0.001)
        self.loss_fn = nn.MSELoss()

    def update(self, states, reward, stds):
        self.optimizer.zero_grad()

        reward_clone = reward.clone().detach().requires_grad_(True)
        predicted_stds = self.network(states)

        reward_clone = reward.clone().detach().requires_grad_(True)
        stds_clone = stds.clone().detach().requires_grad_(True)
        # Compute the loss for the reward proportions
        overall_reward = torch.sum(reward_clone)
        reward_loss = -overall_reward

        # Compute the loss for the stds
        std_loss = ((stds_clone - predicted_stds) ** 2).mean()

        mse_loss = nn.MSELoss()
        # Combine the losses with a weighting factor
        loss = reward_loss + 0.01 * std_loss

        loss.backward()
        self.optimizer.step()

        return predicted_stds, loss

    def act(self, x):
        return self.network(x)
