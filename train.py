import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


import networks
import dataset

from tqdm import tqdm
import numpy as np
from torchvision import transforms, utils, datasets
import pdb
import torch.optim as optim

from mnist import MNIST


net = networks.Net()
net.load_state_dict(torch.load("network.pt"))
net.eval()
net = net.cuda()

train_data = dataset.DigitsDataset(train=True)
test_data = dataset.DigitsDataset(train=False)

objective = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=1e-4)

train_loader = DataLoader(train_data, 
                         batch_size=10,
                         pin_memory=True)

num_epochs = 1
# print(train_data.__getitem__(0))

for epoch in range(num_epochs):
    print("epoch:", epoch+1)
    loop = tqdm(total=len(train_loader), position=0)
    losses = []
    correct = 0
    counted = 0
    for images, labels in train_loader:
        images, labels = images.cuda(async=True), labels.cuda(async=True)
        counted += len(images)
        output = net(images)
        indices = [torch.argmax(sample) for sample in output]
        for l, i in zip(labels, indices):
            if l.item() == i.item():
                correct += 1

        optimizer.zero_grad()

        loss = objective(output, labels)
        losses.append(loss.item())

        loop.set_description('loss: {:.4f} -- Accuracy: {:.2f}%'.format(sum(losses) / len(losses), correct/counted*100))

        loss.backward()

        optimizer.step()
        loop.update(1)
    loop.close()

torch.save(net.state_dict(), "network.pt")
