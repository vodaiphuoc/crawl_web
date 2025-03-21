import torch
import torch.nn as nn
from torch.utils.data import Dataset
from typing import Union, Tuple
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np

class LSTMCellEventContext(nn.Module):
    r"""
    A modify version of Pytorch's nn.LSTMCell
    """
    def __init__(self, 
                 input_price_dim:int = 4, 
                 cell_hidden_dim:int = 768
        )->None:
        r"""
        Args:
            input_price_dim (int): dimension of price vector, stand 
                for (High, Low, Open, Close)
            hidden_dim (int): hidden dimension, equal output features 
                dimension from PhoBert's pooler (768)
        """
        super().__init__()
        self.cell_hidden_dim = cell_hidden_dim
        self.lstm_cell = nn.LSTMCell(input_price_dim, cell_hidden_dim, bias= True)

    def forward(self, 
                batch_price: torch.Tensor, 
                batch_event: torch.Tensor,
                init_hidden_state: Union[torch.Tensor, None],
                init_cell_state: torch.Tensor
        )->Tuple[torch.Tensor]:
        r"""
        Forward process of a single Cell in entire sequence
        Args:
            batch_price (torch.Tensor): input x vector shape (batch, input_price_dim) 
                for (High, Low, Open, Close)
            batch_event (torch.Tensor): output features from PhoBert's 
                pooler shape (batch, 768)
            init_hidden_state (Union[torch.Tensor, None]): last hidden state
                from previous `LSTMCellEventContext` in the sequence
            init_cell_state (Union[torch.Tensor, None]): last cell state
                from previous `LSTMCellEventContext` in the sequence
        Returns:
            a tuple of next hidden state and next cell state
        """
        if init_hidden_state is None:
            hx = batch_event
        else:
            hx = torch.sum(batch_event + init_hidden_state)
        
        return self.lstm_cell(batch_price, (hx, init_cell_state))


class LSTMModel(nn.Module):
    r"""
    A modify of normal Pytorch's LSTM layer which contains
    a list of  `LSTMCellEventContext`
    """
    def __init__(self, 
                 input_price_dim:int = 4, 
                 cell_hidden_dim:int = 768,
                 sequence_length: int = 20,
                 final_output_dim: int = 1
        )->None:
        super().__init__()
        self.cell_hidden_dim = cell_hidden_dim
        self.lstm_cell_list = nn.ModuleList([
            LSTMCellEventContext(
                input_price_dim = input_price_dim, 
                cell_hidden_dim = cell_hidden_dim
            )
            for _ in range(sequence_length)
        ])


        self.fc = nn.Linear(cell_hidden_dim, final_output_dim)

    def forward(self, 
                batch_price: torch.Tensor, 
                batch_event: torch.Tensor
        )->torch.Tensor:
        r"""
        Forward method for processing input sequence
        Args:
            batch_price (torch.Tensor): input x vector 
                shape (batch, sequence_length , input_price_dim)
            batch_event (torch.Tensor): event embedding vector 
                shape (batch, sequence_length , cell_hidden_dim)
        """
        batch_size = batch_price.size(0)
        
        for _ith, _lstm_cell in enumerate(self.lstm_cell_list):
            next_hx, next_cx = _lstm_cell(batch_price = batch_price[:,_ith,:], 
                batch_event = batch_event[:,_ith,:],
                init_hidden_state = None if _ith == 0 else next_hx,
                init_cell_state = torch.zeros(size = (batch_size, self.cell_hidden_dim)) if _ith == 0 else next_cx
            )


        # We are interested in the output at the last time step for prediction
        predicted_price = self.fc(next_hx)
        return predicted_price



class MergeDataset(Dataset):

    _output_dtype = torch.float32

    def __init__(self, 
                 sequence_length:int, 
                 datadf: pd.DataFrame
        )->None:
        super().__init__()
        self.sequence_length = sequence_length
        
        self.df = datadf
    
        # convert merge corpus to embedding vetors
        self.sentence_model = SentenceTransformer(
            'dangvantuan/vietnamese-document-embedding',
            cache_folder= ".checkpoint",
            trust_remote_code=True
        )

        self.embedding_dim = self.sentence_model.get_sentence_embedding_dimension()

        self.sentence_model.compile(fullgraph = True)

    def __len__(self):
        return len(self.df) - (self.sequence_length+1)
    
    def __getitem__(self, index:int):
        row = self.df.loc[index: index + self.sequence_length - 1,:]

        target_price = torch.tensor(
            self.df.loc[index + self.sequence_length,:].close, 
            dtype= self._output_dtype
        )

        price_vector = torch.from_numpy(
            np.concatenate([row.high.to_numpy().reshape(-1,1), 
                            row.low.to_numpy().reshape(-1,1), 
                            row.open.to_numpy().reshape(-1,1), 
                            row.close.to_numpy().reshape(-1,1)
                            ], axis=-1
        )).to(self._output_dtype)

        corpus = row.merge_corpus.copy()
        corpus.reset_index(drop= True, inplace= True)

        check_null_df = corpus.isnull()
        null_ids = corpus[check_null_df==True].index.tolist()
        non_null_ids = corpus[check_null_df==False].index.tolist()

        if len(null_ids) == self.sequence_length: 
            return (
                price_vector, 
                torch.zeros(
                    size = (self.sequence_length,self.embedding_dim), 
                    dtype= self._output_dtype
                ),
                target_price
            )
        
        elif len(null_ids) == 0:
            event_embedding = self.sentence_model.encode(
                corpus.tolist(), 
                show_progress_bar = False, 
                precision = 'float32', 
                convert_to_tensor = True
            ).cpu()

            return price_vector, event_embedding.to(self._output_dtype), target_price

        else:
            total_embeddings = np.zeros(
                shape = (self.sequence_length, self.embedding_dim), 
                dtype = np.float32
            )
        
            non_null_event_embedding = self.sentence_model.encode(
                corpus.loc[non_null_ids].tolist(), 
                show_progress_bar = False, 
                precision = 'float32', 
                convert_to_tensor = True
            ).cpu().numpy()

            total_embeddings[non_null_ids,:] = non_null_event_embedding
            
            return price_vector, torch.from_numpy(total_embeddings), target_price