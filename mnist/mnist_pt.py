from typing import Dict, Any

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.dense1 = nn.Linear(784, 512)
        self.dense2 = nn.Linear(512, 10)
        self.relu = nn.ReLU()
        self.loss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.parameters(), lr=0.001)

    def forward(self, x):
        x = self.dense1(x)
        x = self.relu(x)
        x = self.dense2(x)
        return x


def train_and_log_step(params: Dict):
    batch_size: int = params["batch_size"]
    epochs: int = params["epochs"]
    learning_rate: float = params["learning_rate"]

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.view(-1)),
        ]
    )

    train_loader: DataLoader[Any] = torch.utils.data.DataLoader(
        datasets.MNIST("./data", train=True, transform=transform, download=True),
        batch_size=batch_size,
        shuffle=True,
    )
    test_loader: DataLoader[Any] = torch.utils.data.DataLoader(
        datasets.MNIST("./data", train=False, transform=transform, download=True),
        batch_size=batch_size,
        shuffle=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model = Net().to(device)
    criterion: CrossEntropyLoss = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        model.train()
        running_loss: float = 0.0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits = model(x_batch)
            loss: CrossEntropyLoss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}], Loss: {avg_loss:.4f}")

    model.eval()
    correct = 0
    total = 0
    test_loss = 0.0
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            test_loss += loss.item()
            preds = logits.argmax(dim=1)
            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)

    test_acc = correct / total
    avg_test_loss = test_loss / len(test_loader)
    print(f"Test Loss: {avg_test_loss:.4f}, Accuracy: {test_acc:.4f}")


params = {"batch_size": 64, "learning_rate": 0.01, "epochs": 5}

if __name__ == "__main__":
    train_and_log_step(params)
