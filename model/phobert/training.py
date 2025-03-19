import torch.optim as optim
import torch.nn as nn
import torch

from .modeling import LSTMModel
from .config import TrainingConfig


def training(config = TrainingConfig()):


    model = LSTMModel()

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    # Training loop (simplified)
    epochs = config.epochs
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model()
        loss = criterion(outputs, )
        loss.backward()
        optimizer.step()
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

    