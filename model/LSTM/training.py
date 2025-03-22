import torch.optim as optim
import torch.nn as nn
import torch
from tqdm import tqdm
import pandas as pd
from .modeling import LSTMModel, MergeDataset
from .config import TrainingConfig
from .utils import Report

def train(config = TrainingConfig()):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    total_df = pd.read_csv(config.csv_path)

    # train test split
    test_length = int(len(total_df)*config.test_ratio)
    
    train_df = total_df.iloc[:(len(total_df) - test_length),:]
    train_df.reset_index(drop= True, inplace=True)
    test_df = total_df.iloc[(len(total_df) - test_length):,:]
    test_df.reset_index(drop= True, inplace=True)
    
    model = LSTMModel(**config.model.model_dump()).to(torch.float32).to(device)

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

    print('train length: ', len(train_dataset))
    print('test length: ', len(test_dataset))

    train_loader = torch.utils.data.DataLoader(
        dataset = train_dataset, 
        batch_size = config.batch_size,
        shuffle= True,
        drop_last= True
    )

    test_loader = torch.utils.data.DataLoader(
        dataset = test_dataset, 
        batch_size = config.batch_size, 
        shuffle= False,
        drop_last= False
    )

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    
    # Training loop
    epochs = config.epochs
    for epoch in range(epochs):
        model.train()

        mean_train_loss = 0.0
        for prices, event_embeddings, target_price in tqdm(train_loader, total= len(train_loader)):
            
            optimizer.zero_grad()

            prices = prices.to(device)
            event_embeddings = event_embeddings.to(device)
            target_price = target_price.to(device)

            price_outputs = model(
                batch_price = prices, 
                batch_event = event_embeddings
            )

            price_outputs = torch.squeeze(price_outputs, dim= -1)

            loss = criterion(price_outputs, target_price)
            loss.backward()
            optimizer.step()

            mean_train_loss += loss.item()
        
        print(f'Epoch [{epoch+1}/{epochs}], train loss: {mean_train_loss/len(train_loader)}')

        if epoch % 10 == 0 or epoch == epochs - 1:
            total_val_target_price = []
            total_val_predict_price = []
            model.eval()

            mean_val_loss = 0.0
            with torch.no_grad():
                for prices, event_embeddings, target_price in test_loader:

                    prices = prices.to(device)
                    event_embeddings = event_embeddings.to(device)
                    target_price = target_price.to(device)

                    price_outputs = model(
                        batch_price = prices, 
                        batch_event = event_embeddings
                    )

                    price_outputs = torch.squeeze(price_outputs, dim= -1)

                    loss = criterion(price_outputs, target_price)
                    mean_val_loss += loss.item()


                    # collector
                    rescaled_target = train_dataset.price_stats.re_scale_close(
                        scaled_value = target_price.cpu().numpy()
                    ).tolist()
                    rescaled_predict = train_dataset.price_stats.re_scale_close(
                        scaled_value = price_outputs.cpu().numpy()
                    ).tolist()
                    total_val_target_price.extend(rescaled_target)
                    total_val_predict_price.extend(rescaled_predict)
            
            print(f'Epoch [{epoch+1}/{epochs}], val loss: {mean_val_loss/len(test_loader)}')
            Report(target= total_val_target_price, predict= total_val_predict_price, epoch = epoch)

    torch.save(model.state_dict(), "model.pt")