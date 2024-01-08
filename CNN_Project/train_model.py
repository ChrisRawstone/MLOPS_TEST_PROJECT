import logging

import matplotlib.pyplot as plt
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from rich.logging import RichHandler
from torch.utils.data import DataLoader

import wandb
from CNN_Project.data.make_dataset import CorruptMNISTDataset
from CNN_Project.models.model import MyNeuralNet



torch.manual_seed(42)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

rich_handler = RichHandler(markup=True)
rich_handler.setFormatter(logging.Formatter("%(message)s"))  # minimal formatter
logger.addHandler(rich_handler)


def get_accuracy(model, test_loader, device):
    # evaluate model
    model.eval()

    # get accuracy
    correct = 0
    total = 0
    with torch.no_grad():
        for image, label in test_loader:
            image, label = image.to(device), label.to(device)
            y_hat = model(image)
            _, predicted = torch.max(y_hat.data, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()

    return 100 * correct / total


def train_model(train_loader, model, device, wandb_enabled):
    # Get our data
    if wandb_enabled:
        wandb.init(project="cnn-project", entity="Rawstone")

    train_loader = torch.load("data/processed/train_loader.pth")

    model = MyNeuralNet(1, 10).to(device)

    trainer = Trainer(max_epochs=1)
    trainer.fit(model, train_loader)

    return model


if __name__ == "__main__":
    # Get our data
    train_loader = torch.load("data/processed/train_loader.pth")
    test_loader = torch.load("data/processed/test_loader.pth")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MyNeuralNet(1, 10).to(device)

    model = train_model(train_loader, test_loader, model)

    # evaluate model
    model.eval()

    # get accuracy
    accuracy = get_accuracy(model, test_loader)

    print(f"Accuracy: {accuracy:.2f}%")

    torch.save(model.state_dict(), "models/model.pt")
