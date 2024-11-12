# trainer.py
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

class LightningModuleReg(pl.LightningModule):
    def __init__(self, cfg):
        super(LightningModuleReg, self).__init__()
        self.cfg = cfg
        self.model = get_model(cfg.Model)
        self.loss = torch.nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = self.loss(logits, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        val_loss = self.loss(logits, y)
        self.log("val_loss", val_loss)
        return val_loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.cfg.Optimizer.optimizer.params["lr"])

    def train_dataloader(self):
        X_train, y_train = torch.rand(100, 10), torch.randint(0, 5, (100,))  # Datos ficticios
        train_dataset = TensorDataset(X_train, y_train)
        return DataLoader(train_dataset, batch_size=32)

    def val_dataloader(self):
        X_val, y_val = torch.rand(20, 10), torch.randint(0, 5, (20,))  # Datos ficticios
        val_dataset = TensorDataset(X_val, y_val)
        return DataLoader(val_dataset, batch_size=32)
