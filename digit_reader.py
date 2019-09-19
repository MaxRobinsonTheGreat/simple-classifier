import torch
import cv2 as cv
import networks

image = cv.imread("number.png")[:,:,1] / 255

net = networks.Net()
net = net.cuda()
net.load_state_dict(torch.load("network.pt"))
net.eval()

image = torch.tensor(image, dtype=torch.float, device="cuda")
image = image.view(1, 1, 28, 28)

print(torch.argmax(net(image)).item())