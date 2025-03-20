from dataclasses import dataclass, field
import os



@dataclass
class ModelConfig:
    input_price_dim: int = field(default= 4, metadata = "price dimension includes (High, Low, Open, Close)"), 
    cell_hidden_dim: int = field(default= 768, metadata= "hidden dimension for each LSTM cell"),
    sequence_length: int = field(default= 20, metadata= "sequence length of input data"),
    final_output_dim: int = field(default= 1, metadata= "ouput model dimension")

@dataclass
class TrainingConfig:
    csv_path:str = field(default = __file__.replace(
            os.path.join("model","LSTM","config.py"), 
            os.path.join("stage_4_data","total.csv")
    ))
    sequence_length: int = field(default= 20, metadata= "sequence length of input data")
    model: ModelConfig = field(default= ModelConfig(sequence_length= sequence_length), metadata= "Model config params")

    batch_size:int = field(default = 16)
    learning_rate: float = field(default= 0.001)
    epochs: float = field(default = 2, metadata= "Number of epochs for training")