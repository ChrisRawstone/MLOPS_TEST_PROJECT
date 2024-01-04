import torch
from models.model import MyNeuralNet
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from data.make_dataset import CorruptMNISTDataset


torch.manual_seed(42)


def load_files():
    train_image_files = torch.load("data/processed/train_image_files.pth")
    train_target_files = torch.load("data/processed/train_target_files.pth")
    test_image_file = torch.load("data/processed/test_image_file.pth")
    test_target_file = torch.load("data/processed/test_target_file.pth")

    return train_image_files, train_target_files, test_image_file, test_target_file


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

    optimizer = torch.optim.Adam(model.parameters(), lr=0.00045)
    loss_fn = torch.nn.CrossEntropyLoss()

    loss_list = []
    print("test")

    for epoch in range(10):
        for i, (image, label) in enumerate(train_loader):
            image, label = image.to(device), label.to(device)

            optimizer.zero_grad()
            y_hat = model(image)
            loss = loss_fn(y_hat, label)
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch}, Loss: {loss.item()}")
        loss_list.append(loss.item())

    plt.plot(loss_list)
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
