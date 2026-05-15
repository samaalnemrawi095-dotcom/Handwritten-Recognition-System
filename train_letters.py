import os
import torch
import torch.nn as nn
import torch.optim as optim

from PIL import Image
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from letter_model import LetterModel


SAVE_LETTER_MODEL_PATH = "./checkpoint/letter_model.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


transform = transforms.Compose([
    transforms.Lambda(
        lambda img: img.rotate(-90, expand=True).transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    ),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


train_dataset = datasets.EMNIST(
    root="./data",
    split="letters",
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.EMNIST(
    root="./data",
    split="letters",
    train=False,
    download=True,
    transform=transform
)


# EMNIST letters labels are from 1 to 26
# We convert them to 0 to 25
train_dataset.targets = train_dataset.targets - 1
test_dataset.targets = test_dataset.targets - 1


train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


model = LetterModel().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 20


for epoch in range(epochs):
    model.train()
    total_loss = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)

    print(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}")


model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()


accuracy = 100 * correct / total
print(f"Letter Model Accuracy: {accuracy:.2f}%")


os.makedirs("./checkpoint", exist_ok=True)
torch.save(model.state_dict(), SAVE_LETTER_MODEL_PATH)

print("Letter model saved successfully.")