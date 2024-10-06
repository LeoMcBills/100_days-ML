import torch.nn as nn
from resnet import ResNet, ResidualBlock
import torch
from data import device
from data import train_loader

num_classes = 10
num_epochs = 10
batch_size = 16
learning_rate = 0.01

model = ResNet(ResidualBlock, [3, 4, 6, 3]).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr==learning_rate, weight_decay=0.001, momentum=0.9)

# Train the model
total_step = len(train_loader)

print(model)