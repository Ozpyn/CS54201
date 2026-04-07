# Write a CNN program using Tensorflow to recognize a house or a dog.

# Dog data: https://data.mendeley.com/datasets/v5j6m8dzhv/1
# House data: https://images.cv/download/house/374

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image
import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.fc = nn.Linear(32*16*16, 2)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

def train(epochs = 8, batch_size = 32, lr = 0.0005):
    train_data = datasets.ImageFolder("CNN-data/train", transform=transform)
    val_data = datasets.ImageFolder("CNN-data/val", transform=transform)

    train_loader = DataLoader(train_data, batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size)

    model = CNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr)

    wrong = []

    for epoch in range(epochs):
        model.train()
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(imgs), labels)
            loss.backward()
            optimizer.step()
        
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                _, pred = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (pred == labels).sum().item()
        wrong.append(total - correct)
        print(f"Epoch {epoch+1}, Accuracy: {100*correct/total:.2f}%")

    torch.save(model.state_dict(), "house_dog_cnn.pth")
    print("Saved model as house_dog_cnn.pth")

    plt.title("Error over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("# incorrectly guessed")

    plt.plot(wrong, 'o')

    plt.show()

def test(test_image_path):
    test_model = CNN().to(device)
    test_model.load_state_dict(torch.load("house_dog_cnn.pth", map_location=device))
    test_model.eval()

    image = Image.open(test_image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = test_model(image)
        _, predicted = torch.max(output, 1)

    classes = ["dog", "house"]
    print(f"Predicted class for {test_image_path}: {classes[predicted.item()]}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-train", action="store_true")
    parser.add_argument("-test", type=str)
    args = parser.parse_args()

    if args.train:
        train()
    elif args.test:
        test(args.test)
    else:
        print("Usage: python3 bonus.py -train \nOR \npython3 bonus.py -test <image_path>")
        sys.exit(1)