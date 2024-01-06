import torch
from models.model import MyNeuralNet
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from data.make_dataset import CorruptMNISTDataset
import wandb
import logging
from rich.logging import RichHandler
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

wandb.login(key="cc9eaf6580b2ef9ef475fc59ba669b2de0800b92")
wandb.init(project="cnn-project", entity="Rawstone")

torch.manual_seed(42)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

rich_handler = RichHandler(markup=True)
rich_handler.setFormatter(logging.Formatter("%(message)s"))  # minimal formatter
logger.addHandler(rich_handler)



if __name__ == "__main__":

    # Get our data
    train_loader = torch.load('data/processed/train_loader.pth')
    test_loader = torch.load('data/processed/test_loader.pth')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        print(torch.cuda.get_device_name())
    else:
        print("Running on the CPU")
    

    model = MyNeuralNet(1, 10).to(device)


    
    # checkpoint_callback = ModelCheckpoint(dirpath="./models", monitor="val_loss", mode="min")
    # early_stopping_callback = EarlyStopping(monitor="val_loss", patience=3, verbose=True, mode="min")
    

    trainer = Trainer(max_epochs=10)
    trainer.fit(model, train_loader)

    # for epoch in range(10):
    #     for i, (image, label) in enumerate(train_loader):
    #         image, label = image.to(device), label.to(device)

    #         optimizer.zero_grad()
    #         y_hat = model(image)
            
    #         loss = loss_fn(y_hat, label)
    #         loss.backward()
    #         optimizer.step()

    #     logger.info(f"Epoch {epoch}, Loss: {loss.item()}")
    #     wandb.log({"epoch": epoch,"loss": loss.item()})
    #     loss_list.append(loss.item())

    # plt.plot(loss_list)
    # plt.show()

    plt.savefig("reports/figures/loss.png")

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
    print(f"Accuracy of the network on the {total} test images: {100 * correct / total} %")

    torch.save(model, "models/model.pt")

