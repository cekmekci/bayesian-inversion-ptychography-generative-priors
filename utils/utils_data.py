import math
import torch

from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms


class OffsetAddedDataset(Dataset):
    """
    A custom dataset wrapper that applies an offset to the images in an existing dataset.

    This class takes a dataset and applies a fixed offset to each image in the dataset,
    normalizing the images by dividing by the sum of 1 and the offset. This can be useful
    for augmenting image data in a way that modifies the pixel values slightly.

    Attributes:
        dataset (Dataset): The original dataset containing the images.
        eps (float): The offset value to be added to each image. Default is 0.2.

    Methods:
        __len__(): Returns the total number of samples in the dataset.
        __getitem__(idx): Retrieves the sample at the given index, applies the offset
                          and normalization, and returns the modified image along with
                          an empty list (to match the output format of other datasets).
    """
    def __init__(self, dataset, offset = 0.2):
        self.dataset = dataset
        self.eps = offset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image = (self.dataset[idx][0] + self.eps) / (1 + self.eps)
        return image, []


class TrainingDatasetMNIST(Dataset):
    """
    A custom dataset class for preparing the MNIST dataset for training with optional image resizing
    and pixel value offset.

    This class downloads and preprocesses the MNIST dataset, resizing the images to the specified
    dimensions and applying a fixed offset to each image. The processed dataset can then be used
    directly for training models.

    Attributes:
        dataset_path (str): Path to the directory where the MNIST dataset will be stored or loaded from.
        image_size (int): The size to which each MNIST image will be resized. Default is 64.
        offset (float): The offset value to be added to each image before normalizing. Default is 0.2.
        transforms (transforms.Compose): The composed transformation applied to the images, including resizing and conversion to tensors.
        dataset (OffsetAddedDataset): The processed MNIST dataset with applied transformations and offset.

    Methods:
        __len__(): Returns the total number of samples in the dataset.
        __getitem__(idx): Retrieves the sample at the given index, returns the processed image along
                          with an empty list (to match the output format of other datasets).
    """
    def __init__(self, dataset_path, image_size = 64, offset = 0.2):
        self.dataset_path = dataset_path
        self.image_size = image_size
        self.offset = offset
        self.transforms = transforms.Compose([transforms.Resize(image_size), transforms.ToTensor()])
        self.dataset = datasets.MNIST(self.dataset_path, train = True, download = True, transform = self.transforms)
        self.dataset = OffsetAddedDataset(self.dataset, self.offset)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image = self.dataset[idx][0]
        return image, []


class TestDatasetMNIST(Dataset):
    """
    A custom dataset class for preparing the MNIST dataset for testing, with specific image processing
    that involves offset addition, and the generation of complex-valued images from magnitude and phase pairs.

    This class downloads and preprocesses the MNIST dataset for testing, resizing the images, applying a fixed offset,
    and then constructing complex-valued images using magnitude and phase information. The final output consists of
    real and imaginary parts concatenated along the channel dimension.

    Attributes:
        dataset_path (str): Path to the directory where the MNIST dataset will be stored or loaded from.
        image_size (int): The size to which each MNIST image will be resized. Default is 64.
        offset (float): The offset value to be added to each image before normalization. Default is 0.2.
        transforms (transforms.Compose): The composed transformation applied to the images, including resizing and conversion to tensors.
        dataset (OffsetAddedDataset): The processed MNIST dataset with applied transformations and offset, used for generating complex-valued images.

    Methods:
        __len__(): Returns the total number of complex-valued samples in the dataset.
        __getitem__(idx): Retrieves the magnitude and phase image pair at the specified index,
                          constructs the real and imaginary parts, concatenates them, and returns
                          the resulting complex-valued image along with an empty list.
    """
    def __init__(self, dataset_path, image_size = 64, offset = 0.2):
        self.dataset_path = dataset_path
        self.image_size = image_size
        self.offset = offset
        self.transforms = transforms.Compose([transforms.Resize(image_size), transforms.ToTensor()])
        self.dataset = datasets.MNIST(self.dataset_path, train = False, download = True, transform = self.transforms)
        self.dataset = OffsetAddedDataset(self.dataset, self.offset)
        assert len(self.dataset) % 2 == 0

    def __len__(self):
        return len(self.dataset) // 2

    def __getitem__(self, idx):
        # obtain the magnitude and phase images
        magnitude_image = self.dataset[2 * idx][0]
        phase_image = self.dataset[2 * idx + 1][0]
        # obtain the real and imaginary parts
        real_image = magnitude_image * torch.cos(phase_image)
        imag_image = magnitude_image * torch.sin(phase_image)
        # contatenate it along the channel dimension
        image = torch.cat((real_image, imag_image), 0)
        return image, []


def get_mnist_training_dataloader(dataset_path, batch_size = 64, image_size = 64, offset = 0.2):
    """
    Creates and returns a DataLoader for the MNIST training dataset with specified preprocessing options.

    This function initializes a `TrainingDatasetMNIST` instance, which applies transformations including
    resizing, offset addition, and normalization to the MNIST training dataset. It then wraps the dataset
    in a DataLoader to facilitate batch processing during model training.

    Args:
        batch_size (int, optional): The number of samples per batch to load. Default is 64.
        image_size (int, optional): The size to which each MNIST image will be resized. Default is 64.
        offset (float, optional): The offset value to be added to each image before normalization. Default is 0.2.

    Returns:
        DataLoader: A DataLoader instance for the processed MNIST training dataset, with shuffling enabled
                    for randomizing the order of samples in each epoch.
    """
    train_dataset = TrainingDatasetMNIST(dataset_path = dataset_path, image_size = image_size, offset = offset)
    train_dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
    return train_dataloader


def get_mnist_test_dataloader(dataset_path, batch_size = 1, image_size = 64, offset = 0.2):
    """
    Creates and returns a DataLoader for the MNIST test dataset with specified preprocessing options.

    This function initializes a `TestDatasetMNIST` instance, which applies transformations including
    resizing, offset addition, and the creation of complex-valued images from magnitude and phase pairs
    to the MNIST test dataset. It then wraps the dataset in a DataLoader to facilitate batch processing during model evaluation.

    Args:
        batch_size (int, optional): The number of samples per batch to load. Default is 1.
        image_size (int, optional): The size to which each MNIST image will be resized. Default is 64.
        offset (float, optional): The offset value to be added to each image before normalization. Default is 0.2.

    Returns:
        DataLoader: A DataLoader instance for the processed MNIST test dataset, with shuffling disabled
                    to maintain the order of samples during evaluation.
    """
    test_dataset = TestDatasetMNIST(dataset_path = dataset_path, image_size = image_size, offset = offset)
    test_dataloader = DataLoader(test_dataset, batch_size = batch_size, shuffle = False)
    return test_dataloader


if __name__ == '__main__':

    # test the training dataset
    training_dataset = TrainingDatasetMNIST(dataset_path = "./data/", image_size = 64, offset = 0.2)
    training_example, _ = training_dataset[0]
    print("Shape of a training example:", training_example.shape)

    # test the test dataset
    test_dataset = TestDatasetMNIST(dataset_path = "./data/", image_size = 64, offset = 0.2)
    test_dataset, _ = test_dataset[0]
    print("Shape of a test example:", test_dataset.shape)

    # test the training dataloader
    train_dataloader = get_mnist_training_dataloader(dataset_path = "./data/", batch_size = 64, image_size = 64, offset = 0.2)
    train_minibatch_images, _ = next(iter(train_dataloader))
    print("Minibatch of training images:", train_minibatch_images.shape)

    # test the training dataloader
    test_dataloader = get_mnist_test_dataloader(dataset_path = "./data/", batch_size = 64, image_size = 64, offset = 0.2)
    test_minibatch_images, _ = next(iter(test_dataloader))
    print("Minibatch of test images:", test_minibatch_images.shape)
