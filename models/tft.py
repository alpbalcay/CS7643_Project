import torch
from darts.models import TFTModel as TFTModeL
from pytorch_lightning.callbacks import EarlyStopping


class TFTModel:
    def __init__(self, batch_size, lr,
                 n_epoch, hidden_size, lstm_layers, 
                 n_head, full_attention, dropout,
                 hidden_continuous_size):
        self.batch_size = batch_size
        self.epoch = n_epoch
        self.hidden_size = hidden_size
        self.lstm_layers = lstm_layers
        self.n_head = n_head
        self.full_attention = full_attention
        self.hidden_continuous_size = hidden_continuous_size
        self.dropout = dropout
        self.lr = lr

    def build(self, input_chunk_length, output_chunk_length):
        torch.manual_seed(42)

        callbacks = [EarlyStopping("val_loss", min_delta=0.0001, patience=2, verbose=True)]

        pl_trainer_kwargs = {
            "accelerator": "gpu" if torch.cuda.is_available() else None,
            "callbacks": callbacks
        }

        model = TFTModeL(
            input_chunk_length=input_chunk_length,
            output_chunk_length=output_chunk_length,
            hidden_size=self.hidden_size,
            lstm_layers=self.lstm_layers,
            num_attention_heads=self.n_head,
            full_attention=self.full_attention,
            hidden_continuous_size = self.hidden_continuous_size,
            dropout=self.dropout,
            batch_size=self.batch_size,
            n_epochs=self.epoch,
            add_encoders=None,
            likelihood=None, 
            loss_fn=torch.nn.MSELoss(),
            random_state=42,
            optimizer_kwargs={"lr": self.lr},
            pl_trainer_kwargs=pl_trainer_kwargs,
            model_name="tft_model",
            force_reset=True,
            save_checkpoints=True,
        )

        return model