import torch.optim as optim
import torch.nn as nn
import torch
import pandas as pd
from .modeling import LSTMModel, MergeDataset
from .config import TrainingConfig
from .utils import Report

def train(config = TrainingConfig()):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    total_df = pd.read_csv(config.csv_path)
    
    # for exp only
    total_df = total_df.head(100)

    # train test split
    test_length = int(len(total_df)*config.test_ratio)
    
    train_df = total_df.iloc[:(len(total_df) - test_length),:]
    train_df.reset_index(drop= True, inplace=True)
    test_df = total_df.iloc[(len(total_df) - test_length):,:]
    test_df.reset_index(drop= True, inplace=True)
    

    model = LSTMModel().to(torch.float32).to(device)

    train_dataset = MergeDataset(
        sequence_length = config.sequence_length, 
        datadf = train_df,
        scale_by_other= False,
        other_price_stats= None
    )

    test_dataset = MergeDataset(
        sequence_length = config.sequence_length, 
        datadf = test_df,
        scale_by_other= True,
        other_price_stats= train_dataset.price_stats
    )

    train_loader = torch.utils.data.DataLoader(
        dataset = train_dataset, 
        batch_size = config.batch_size, 
        shuffle= True
    )

    test_loader = torch.utils.data.DataLoader(
        dataset = test_dataset, 
        batch_size = config.batch_size, 
        shuffle= False
    )

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    
    # Training loop
    epochs = config.epochs
    for epoch in range(epochs):
        model.train()
        for prices, event_embeddings, target_price in train_loader:
            
            optimizer.zero_grad()

            prices = prices.to(device)
            event_embeddings = event_embeddings.to(device)
            target_price = target_price.to(device)

            price_outputs = model(
                batch_price = prices, 
                batch_event = event_embeddings
            )

            loss = criterion(price_outputs, target_price)
            loss.backward()
            optimizer.step()
            print(f'Epoch [{epoch+1}/{epochs}], train loss: {loss.item():.4f}')

        if epoch % 10 == 0 or epoch == epochs - 1:
            total_val_target_price = []
            total_val_predict_price = []
            model.eval()
            with torch.no_grad():
                for prices, event_embeddings, target_price in test_loader:

                    prices = prices.to(device)
                    event_embeddings = event_embeddings.to(device)
                    target_price = target_price.to(device)

                    price_outputs = model(
                        batch_price = prices, 
                        batch_event = event_embeddings
                    )

                    loss = criterion(price_outputs, target_price)
                    print(f'Epoch [{epoch+1}/{epochs}], val loss: {loss.item():.4f}')


                    # collector
                    rescaled_target = train_dataset.price_stats.re_scale_close(
                        scaled_value = target_price.cpu().numpy()
                    ).tolist()
                    rescaled_predict = train_dataset.price_stats.re_scale_close(
                        scaled_value = price_outputs.cpu().numpy()
                    ).tolist()
                    total_val_target_price.extend(rescaled_target)
                    total_val_predict_price.extend(rescaled_predict)

            Report(target= total_val_target_price, predict= total_val_predict_price, epoch = epoch)