import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from .nn import MetaNet


def train_nn_from_replay_buffer(replay_buffer, state_dim, action_dim, hidden_dim=128, batch_size=64, learning_rate=0.001, num_epochs=100):


    input_data = torch.tensor([(s,a) for s, a, _ in replay_buffer], dtype=torch.float32)
    next_states = torch.tensor([s_ for _, _, s_ in replay_buffer], dtype=torch.float32)

    # Create a DataLoader to handle mini-batches during training
    dataset = TensorDataset(input_data, next_states)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize the neural network
    net = MetaNet(state_dim, action_dim, hidden_dim)

    # Define the loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        for batch_inputs, batch_next_states in data_loader:
            # Zero the gradients of the optimizer
            optimizer.zero_grad()

            # Forward pass: compute the predicted next states
            predicted_next_states = net(batch_inputs)

            # Compute the loss between the predicted next states and the actual next states
            loss = criterion(predicted_next_states, batch_next_states)

            # Backward pass: compute gradients and update the neural network parameters
            loss.backward()
            optimizer.step()

        # Print the loss for monitoring the training progress after each epoch
        print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

    # Return the trained neural network
    return net
