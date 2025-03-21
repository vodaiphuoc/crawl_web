import torch.optim as optim
import torch.nn as nn
import torch
import pandas as pd
from .modeling import LSTMModel, MergeDataset
from .config import TrainingConfig


def train(config = TrainingConfig()):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    total_df = pd.read_csv(config.csv_path)

    model = LSTMModel().to(torch.float32).to(device)

    dataset = MergeDataset(sequence_length = config.sequence_length, datadf = total_df)
    loader = torch.utils.data.DataLoader(
        dataset = dataset, 
        batch_size = config.batch_size, 
        shuffle= True
    )

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    # Training loop
    epochs = config.epochs
    for epoch in range(epochs):
        model.train()
        for prices, event_embeddings, target_price in loader:
            
            optimizer.zero_grad()

            prices = prices.to(device)
            event_embeddings = event_embeddings.to(device)
            target_price = target_price.to(device)
            
            print(target_price.shape)

            price_outputs = model(
                batch_price = prices, 
                batch_event = event_embeddings
            )

            loss = criterion(price_outputs, target_price)
            loss.backward()
            optimizer.step()
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
    