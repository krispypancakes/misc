import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
from torchvision import datasets, transforms
import os
from tqdm import tqdm

BS = 128

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((.5), (.5))])

mnist_train = datasets.MNIST('data', train=True, download=True, transform=transform)
mnist_test = datasets.MNIST('data', train=False, download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(mnist_train, batch_size=BS, shuffle=True, pin_memory=True)
testloader = torch.utils.data.DataLoader(mnist_test, batch_size=BS, shuffle=False, pin_memory=False)


class MnistClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5), nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.MaxPool2d(2))
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(2))
        self.fc1 = nn.Linear(in_features=64*4*4, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=10) # 10 classes

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = F.dropout(x)
        x = x.view(-1, 64*4*4) # flatten 
        x = F.relu(self.fc1(x))
        return self.fc2(x)
    
Epochs = 2
model = MnistClassifier()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

device = torch.device("mps")
model.to(device)
epoch_bar = tqdm(range(Epochs), desc="training", unit="epoch")

losses = []
for epoch in epoch_bar:
    model.train()
    running_loss = 0.0
    for x, y in trainloader:
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type="mps"):
            y_hat = model(x)
            loss = criterion(y_hat, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    epoch_loss = running_loss / len(trainloader)
    losses.append(epoch_loss)

    epoch_bar.set_postfix({"epoch": epoch, "loss": epoch_loss})
