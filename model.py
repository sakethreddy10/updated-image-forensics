import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torchvision.models import resnet18, ResNet18_Weights

# define the training directory
TRAIN_DIR = "dataset/train"

# transformation for training data
transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# load training datset
train_dataset = datasets.ImageFolder(root=TRAIN_DIR, transform=transforms)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# print basic info
print("Number of training images:", len(train_dataset))
print("Number of training batches:", len(train_loader))
print("Images per batch:", train_loader.batch_size)

# define the model
model = resnet18(weights=ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, 2)

# set the model to device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# set loss fn & optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# define the training function
def train(model, train_loader, criterion, optimizer, device):
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    # Iterate over the training data
    for i, (inputs, labels) in enumerate(train_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Update the running loss and the number of correct predictions
        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # Print the loss and accuracy for this batch
        if (i+1) % 10 == 0:
            batch_loss = running_loss / (i+1) / inputs.size(0)
            batch_accuracy = correct / total
            print(f'Batch {i+1}/{len(train_loader)}, Loss: {batch_loss:.4f}, Accuracy: {batch_accuracy:.4f}')

    # Compute the loss and accuracy for the entire epoch
    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_accuracy = correct / total

    return epoch_loss, epoch_accuracy

# train the model
num_epochs = 10
for epoch in range(num_epochs):
    epoch_loss, epoch_accuracy = train(model, train_loader, criterion, optimizer, device)
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}')

# TODO: evaluate the model

# save the model
torch.save(model.state_dict(), "model.pth")