import torch
from darts.models import RNNModel
from pytorch_lightning.callbacks import EarlyStopping


class LSTMModel:
    def __init__(self, val_len, batch_size, lr,
                 n_epoch, hidden_dim, n_rnn_layers, dropout):
        self.val_len = val_len
        self.batch_size = batch_size
        self.epoch = n_epoch
        self.hidden_dim = hidden_dim
        self.rnn_layers = n_rnn_layers
        self.dropout = dropout
        self.lr = lr

    def build(self, input_chunk_length):
        torch.manual_seed(42)

        callbacks = [EarlyStopping("val_loss", min_delta=0.0001, patience=2, verbose=True)]

        pl_trainer_kwargs = {
            "accelerator": "gpu" if torch.cuda.is_available() else None,
            "callbacks": callbacks
        }

        model = RNNModel(
            model="LSTM",
            input_chunk_length=input_chunk_length,
            hidden_dim=self.hidden_dim,
            n_rnn_layers=self.rnn_layers,
            dropout=self.dropout,
            training_length=input_chunk_length + self.val_len - 1,
            n_epochs=self.epoch,
            batch_size=self.batch_size,
            optimizer_kwargs={"lr": self.lr},
            pl_trainer_kwargs=pl_trainer_kwargs,
            model_name="lstm_model",
            force_reset=True,
            save_checkpoints=True
        )

        return model
