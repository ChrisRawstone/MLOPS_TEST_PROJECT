import torch
from torch.utils.data import DataLoader

from CNN_Project.data.make_dataset import CorruptMNISTDataset, mnist
from CNN_Project.models.model import MyNeuralNet
from CNN_Project.train_model import get_accuracy, train_model
import os.path
import pytest

@pytest.mark.skipif(not os.path.exists("data/raw/corruptmnist"), reason="Data files not found")
def test_make_mnist():
    train_dataset, test_dataset = mnist("data/raw/corruptmnist")

    assert len(train_dataset) == 30000
    assert len(test_dataset) == 5000

    # Check that the first image is of shape (1, 28, 28)
    image, label = train_dataset[0]
    assert image.shape == torch.Size([1, 28, 28])

    # assert that all labels are represented
    assert set(train_dataset.targets.unique().tolist()) == set(range(10))

@pytest.mark.skipif(not os.path.exists("data/processed/train_loader.pth"), reason="Data files not found")
def test_train_model():
    # evaluate model
    train_loader = torch.load("data/processed/train_loader.pth")
    test_loader = torch.load("data/processed/test_loader.pth")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        print(torch.cuda.get_device_name())
    else:
        print("Running on the CPU")

    model = MyNeuralNet(1, 10).to(device)

    model = train_model(train_loader, model, device, wandb_enabled=False)

    # evaluate model
    model.eval()

    # get accuracy
    accuracy = get_accuracy(model, test_loader, device)

    print(f"Accuracy: {accuracy:.2f}%")

    torch.save(model.state_dict(), "models/model.pt")

    assert accuracy > 90.0

@pytest.mark.skipif(not os.path.exists("models/model.pt"), reason="models not found")
def test_prediction_of_test_set():
    model = MyNeuralNet(1, 10)
    model.load_state_dict(torch.load("models/model.pt"))

    # create artificial image
    image = torch.rand(64, 1, 28, 28)

    y_pred = model(image)

    # Recreate DataLoaders
    print(y_pred.shape)

    print(y_pred.shape == torch.Size([64, 10]))
    assert y_pred.shape == torch.Size([64, 10])


if __name__ == "__main__":
    test_train_model()