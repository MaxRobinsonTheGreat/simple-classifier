from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, datasets

class DigitsDataset(Dataset):
    """Handwritten digits train dataset."""

    def __init__(self, train):
        self.train_data = datasets.MNIST("./data", 
                                     train=train, 
                                     transform=transforms.ToTensor(), 
                                     download=True)

    def __len__(self):
        # return 1
        return len(self.train_data)

    def __getitem__(self, idx):
        sample, label = self.train_data[idx]
        return sample, label