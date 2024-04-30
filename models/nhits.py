import torch
from darts.models import NHiTSModel
from pytorch_lightning.callbacks import EarlyStopping


class NHitsModel:
    def __init__(self, batch_size, lr,
                 n_epoch, num_stack, num_blocks, 
                 num_layers, layer_exp, dropout):
        self.batch_size = batch_size
        self.epoch = n_epoch
        self.num_stack = num_stack
        self.num_blocks = num_blocks
        self.num_layers = num_layers
        self.layer_exp = layer_exp
        self.dropout = dropout
        self.lr = lr

    def build(self, input_chunk_length, output_chunk_length):
        torch.manual_seed(42)

        callbacks = [EarlyStopping("val_loss", min_delta=0.0001, patience=2, verbose=True)]

        pl_trainer_kwargs = {
            "accelerator": "gpu" if torch.cuda.is_available() else None,
            "callbacks": callbacks
        }

        model = NHiTSModel(
            input_chunk_length=input_chunk_length,
            output_chunk_length=output_chunk_length,
            num_stacks=self.num_stack,
            num_blocks=self.num_blocks,
            num_layers=self.num_layers,
            layer_widths=2 ** self.layer_exp,
            dropout=self.dropout,
            n_epochs=self.epoch,
            batch_size=self.batch_size,
            add_encoders=None,
            likelihood=None, 
            loss_fn=torch.nn.MSELoss(),
            random_state=42,
            optimizer_kwargs={"lr": self.lr},
            pl_trainer_kwargs=pl_trainer_kwargs,
            model_name="nhits_model",
            force_reset=True,
            save_checkpoints=True,
        )

        return model