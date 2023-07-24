import torch
import torch.nn as nn

class MetaNet(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(MetaNet, self).__init__()

        # Define the hidden layer dimensions
        hidden_layer_1 = hidden_dim
        hidden_layer_2 = hidden_dim

        # Define the layers of the neural network
        self.layers = nn.Sequential(
            nn.Linear(2, hidden_layer_1),
            nn.ReLU(),
            nn.Linear(hidden_layer_1, hidden_layer_2),
            nn.ReLU(),
            nn.Linear(hidden_layer_2, 1)
        )

    def forward(self, X):

        # Pass the input through the neural network
        next_state_prediction = self.layers(X)

        return next_state_prediction.view(-1)
