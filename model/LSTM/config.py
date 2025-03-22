from pydantic import Field, computed_field, BaseModel
import os


class ModelConfig(BaseModel):
    input_price_dim: int =  4
    cell_hidden_dim: int =  768
    last_cfl_hidden_dim: int =  256
    sequence_length: int =  20
    final_output_dim: int =  1

class TrainingConfig(BaseModel):
    csv_path:str = Field(default = __file__.replace(
            os.path.join("model","LSTM","config.py"), 
            os.path.join("stage_4_data","total.csv")
    ))
    sequence_length: int =  20
    batch_size:int = 32
    learning_rate: float =  0.001
    epochs: float = 100
    test_ratio: float = 0.4

    @computed_field
    @property
    def model(self)->ModelConfig:
        return ModelConfig(sequence_length = self.sequence_length)
