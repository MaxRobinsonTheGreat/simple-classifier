import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import torch.optim as optim

from mnist import MNIST 

class TrainDataset(Dataset):
    """Handwritten digits train dataset."""

    def __init__(self):
        self.train_data = MNIST('train_set')
        self.images, self.labels = self.train_data.load_training()

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        return self.labels, torch.tensor(self.images[idx])

class TestDataset(Dataset):
    """Handwritten digits test dataset."""

    def __init__(self):
        self.test_data = MNIST('test_set')
        self.images, self.labels = self.test_data.load_testing()

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        return self.labels, torch.tensor(self.images[idx])

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 24 * 24, 120)  # 6*6 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


net = Net()
train_data = TrainDataset()

label, input = train_data.__getitem__(0)
# print(input.shape)
output = net(input)
# target = torch.randn(10)
# target = target.view(1, -1)
# criterion = nn.MSELoss()

# loss = criterion(output, target)
# print(loss)

# optimizer = optim.SGD(net.parameters(), lr=0.01)

# # in your training loop:
# optimizer.zero_grad()   # zero the gradient buffers
# output = net(input)
# loss = criterion(output, target)
# loss.backward()
# optimizer.step()