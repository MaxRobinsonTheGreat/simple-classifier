import torch
import networks
import dataset

from torch.utils.data import Dataset, DataLoader


net = networks.Net()
net.load_state_dict(torch.load("network.pt"))
net.eval()
net = net.cuda()

test_data = dataset.DigitsDataset(train=False)
test_loader = DataLoader(test_data, 
                         batch_size=10,
                         pin_memory=True)

objective = torch.nn.CrossEntropyLoss()

correct = 0
total_loss = 0
for images, labels in test_loader:
    images, labels = images.cuda(async=True), labels.cuda(async=True)
    output = net(images)
    indices = [torch.argmax(sample) for sample in output]
    for l, i in zip(labels, indices):
        if l.item() == i.item():
            correct += 1
    total_loss += objective(output, labels).item()

print(correct)
print("Test accuracy:", correct/len(test_data)*100, "%")
print("Test average loss:", total_loss/len(test_loader))
