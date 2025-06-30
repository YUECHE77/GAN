import os
import pandas as pd
from PIL import Image

from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, ), std=(0.5, ))  # for gray image -> 3 if it is RGB image
])

class FashionMnist(Dataset):
    def __init__(self, csv_path, transform=None):
        fashion_df = pd.read_csv(csv_path)
        print(fashion_df.shape)
        self.labels = fashion_df.label.values
        self.images = fashion_df.iloc[:, 1:].values.astype('uint8').reshape(-1, 28, 28)

        self.transform = transform
    
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.fromarray(self.images[idx])
        label = self.labels[idx]

        if self.transform is not None:
            image = self.transform(image)
        
        return image, label

def FashionMNIST(csv_path, batch_size, transform=transform):
    dataset = FashionMnist(csv_path, transform=transform)
    dataloader = DataLoader(dataset=dataset,
                            batch_size=batch_size,
                            shuffle=True)
    return dataloader

def MNIST(img_folder, batch_size, transform=transform):
    """
    Return the trining dataloader.
    No need for val/test data, since it is a generative task, the input should always
    be a random latent vector.
    """
    os.makedirs(img_folder, exist_ok=True)

    train_set = torchvision.datasets.MNIST(root=img_folder, train=True,
                                           download=True, transform=transform)

    dataloader = DataLoader(dataset=train_set,
                            batch_size=batch_size,
                            shuffle=True)
    return dataloader

if __name__ == '__main__':
    img_folder = r'/root/autodl-tmp/MNIST/'
    csv_path = r'/root/autodl-tmp/FashionMNIST/fashion-mnist_test.csv'
    batch_size = 16

    MNIST_loader = MNIST(img_folder, batch_size)
    FashionMNIST_loader = FashionMNIST(csv_path, batch_size)

    # Check for MNIST
    images, labels = next(iter(MNIST_loader))
    print('\nFor MNIST:')
    print(f'images shape: {images.shape}')  # [batch_size, 1, 28, 28]
    print(f'labels shape: {labels.shape}\n')  # [batch_size, ]

    # Check for FashionMNIST:
    images, labels = next(iter(FashionMNIST_loader))
    print('For FashionMNIST:')
    print(f'images shape: {images.shape}')  # [batch_size, 1, 28, 28]
    print(f'labels shape: {labels.shape}')  # [batch_size, ]