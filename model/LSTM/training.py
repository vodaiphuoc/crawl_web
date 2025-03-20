import torch.optim as optim
import torch.nn as nn
import torch
import pandas as pd
from .modeling import LSTMModel, MergeDataset
from .config import TrainingConfig


def train(config = TrainingConfig()):
    total_df = pd.read_csv(config.csv_path)

    model = LSTMModel()
    dataset = MergeDataset(sequence_length= config.sequence_length, datadf=total_df)
    loader = torch.utils.data.DataLoader(
        dataset=dataset, 
        batch_size= config.batch_size, 
        shuffle= True
    )
    
    for prices, event_embeddings in loader:
        print(prices.shape)
        print(event_embeddings.shape)

    # # Define loss function and optimizer
    # criterion = nn.MSELoss()
    # optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    # # Training loop (simplified)
    # epochs = config.epochs
    # for epoch in range(epochs):
    #     optimizer.zero_grad()
    #     outputs = model()
    #     loss = criterion(outputs, )
    #     loss.backward()
    #     optimizer.step()
    #     print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

    