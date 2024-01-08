import torch
from torch.utils.data import DataLoader

from CNN_Project.data.make_dataset import CorruptMNISTDataset
from CNN_Project.models.model import MyNeuralNet


def predict(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader) -> None:
    """Run prediction for a given model and dataloader.

    Args:
        model: model to use for prediction
        dataloader: dataloader with batches

    Returns
        Tensor of shape [N, d] where N is the number of samples and d is the output dimension of the model

    """
    predictions = []

    for batch in dataloader:
        image, label = batch
        y = model(image)
        predictions.append(y)

    return torch.cat(predictions, 0)


if __name__ == "__main__":
    # Load model
    model = MyNeuralNet(1, 10)
    model.load_state_dict(torch.load("models/model.pt"))

    # Load datasets
    testLoader = torch.load("data/processed/test_loader.pth")

    # Run prediction
    y_pred = predict(model, testLoader)

    print("Done")
