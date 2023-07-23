import torch
import torch.nn as nn

class MetaNet(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(MetaNet, self).__init__()

        # Define the input dimensions (state_dim + action_dim) for the state-action pair
        input_dim = state_dim + action_dim

        # Define the hidden layer dimensions
        hidden_layer_1 = hidden_dim
        hidden_layer_2 = hidden_dim

        # Define the layers of the neural network
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_layer_1),
            nn.ReLU(),
            nn.Linear(hidden_layer_1, hidden_layer_2),
            nn.ReLU(),
            nn.Linear(hidden_layer_2, state_dim)
        )

    def forward(self, state, action):
        # Concatenate state and action to create the input for the neural network
        input_data = torch.cat([state, action], dim=-1)

        # Pass the input through the neural network
        next_state_prediction = self.layers(input_data)

        return next_state_prediction
