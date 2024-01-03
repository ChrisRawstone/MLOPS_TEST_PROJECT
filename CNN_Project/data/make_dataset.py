import torch
from torch.utils.data import Dataset
from pathlib import Path
import os


class CorruptMNISTDataset(Dataset):
    def __init__(self, image_files, target_files):
        self.images = torch.cat([torch.load(file) for file in image_files])
        self.targets = torch.cat([torch.load(file) for file in target_files])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Assuming each .pt file contains a single image and its corresponding target
        image = self.images[idx].unsqueeze(0).float()  # Ensure image is of type float
        target = self.targets[idx].long()  # Ensure target is of type long for classification
        return image, target


def mnist():
    # Define the path to the dataset files
    data_path = Path(
        "/mnt/c/Users/chris/OneDrive/Dokumenter/Skole/Universitet/Kandidat/1Semester/MachineLearningOperations/dtu_mlops/data/corruptmnist"
    )

    # Define training and testing file names
    train_image_files = sorted(data_path.glob("train_images_*.pt"))
    train_target_files = sorted(data_path.glob("train_target_*.pt"))
    test_image_file = data_path / "test_images.pt"
    test_target_file = data_path / "test_target.pt"

    # Load the dataset files into the custom Dataset
    train_dataset = CorruptMNISTDataset(train_image_files, train_target_files)
    test_dataset = CorruptMNISTDataset([test_image_file], [test_target_file])

    # Create data loaders

    torch.save(train_image_files, "data/processed/train_image_files.pth")
    torch.save(train_target_files, "data/processed/train_target_files.pth")

    torch.save(test_image_file, "data/processed/test_image_file.pth")
    torch.save(test_target_file, "data/processed/test_target_file.pth")

    return train_dataset, test_dataset


if __name__ == "__main__":
    mnist()

    print(os.getcwd())

    # Save datasets
    # torch.save(train_dataset, 'data/processed/train_dataset.pth')
    # torch.save(test_dataset, 'data/processed/test_dataset.pth')
