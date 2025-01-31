import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
from collections import Counter

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Balanced Data Augmentation (Fixed)
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),  # Less aggressive rotation
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),  # Subtle color changes
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

transform_valid = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load datasets
train_dataset = datasets.ImageFolder(root="dataset/rvf10k/train", transform=transform_train)
val_dataset = datasets.ImageFolder(root="dataset/rvf10k/valid", transform=transform_valid)

# DataLoader (Keep num_workers=0 for Windows compatibility)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0)

# Load Pretrained ResNet-18 Model
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

# Fine-tune More Layers
for param in list(model.parameters())[:-6]:  
    param.requires_grad = False  

# Modify Last Layer for Binary Classification
num_features = model.fc.in_features
model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(num_features, 2)
)

model = model.to(device)

# Compute Class Weights
class_counts = Counter(train_dataset.targets)
class_weights = [len(train_dataset) / (2 * count) for count in class_counts.values()]
criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights).to(device))

# Optimizer and Learning Rate
optimizer = optim.Adam(model.parameters(), lr=0.00005)  # Lower LR for stability
lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

# Training Loop
for epoch in range(10):
    print(f"Epoch {epoch + 1}/10")
    
    # Train
    model.train()
    for inputs, labels in tqdm(train_loader, desc="Training"):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    # Validate
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc="Validation"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds) * 100
    print(f"Validation Accuracy: {acc:.2f}%")

