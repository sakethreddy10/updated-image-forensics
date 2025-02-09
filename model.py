import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Advanced Data Augmentation for Training
transform_train = transforms.Compose([
    transforms.Resize((256, 256)),  # Slightly larger resize for better cropping
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),  # Random cropping
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),  # Increased rotation range
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),  # More aggressive color jitter
    transforms.RandomGrayscale(p=0.2),  # Increased grayscale probability
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Random translation
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Simple Transformations for Validation
transform_valid = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load datasets
train_dataset = datasets.ImageFolder(root="dataset/rvf10k/train", transform=transform_train)
val_dataset = datasets.ImageFolder(root="dataset/rvf10k/valid", transform=transform_valid)

# Handle Class Imbalance with WeightedRandomSampler
class_counts = Counter(train_dataset.targets)
class_weights = [1.0 / class_counts[i] for i in range(len(class_counts))]
sample_weights = [class_weights[target] for target in train_dataset.targets]
sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

# DataLoader with WeightedRandomSampler
train_loader = DataLoader(train_dataset, batch_size=64, sampler=sampler, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

# Load Pretrained ResNet-50 Model (Upgraded from ResNet-18)
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

# Fine-tune More Layers
for param in list(model.parameters())[:-20]:  # Unfreeze more layers for better fine-tuning
    param.requires_grad = False

# Modify Last Layer for Binary Classification
num_features = model.fc.in_features
model.fc = nn.Sequential(
    nn.Dropout(0.7),  # Increased dropout for better regularization
    nn.Linear(num_features, 2)
)

model = model.to(device)

# Compute Class Weights for Loss Function
class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)  # Added label smoothing

# Optimizer with L2 Regularization
optimizer = optim.AdamW(model.parameters(), lr=0.00005, weight_decay=1e-5)  # AdamW for better optimization
lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3)

# Training Function
def train(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    y_true, y_pred = [], []

    for inputs, labels in tqdm(train_loader, desc="Training"):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        preds = torch.argmax(outputs, dim=1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

    acc = accuracy_score(y_true, y_pred) * 100
    precision = precision_score(y_true, y_pred, average="weighted", zero_division=1)
    recall = recall_score(y_true, y_pred, average="weighted", zero_division=1)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=1)

    return running_loss / len(train_loader), acc, precision, recall, f1

# Validation Function
def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    y_true, y_pred = [], []

    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc="Validation"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    acc = accuracy_score(y_true, y_pred) * 100
    precision = precision_score(y_true, y_pred, average="weighted", zero_division=1)
    recall = recall_score(y_true, y_pred, average="weighted", zero_division=1)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=1)
    cm = confusion_matrix(y_true, y_pred)

    return running_loss / len(val_loader), acc, precision, recall, f1, cm, y_true, y_pred

# Train Model & Save Best Version
best_val_acc = 0.0
for epoch in range(20):  # Increased number of epochs
    print(f"Epoch {epoch + 1}/20")

    train_loss, train_acc, train_precision, train_recall, train_f1 = train(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc, val_precision, val_recall, val_f1, val_cm, y_true, y_pred = validate(model, val_loader, criterion, device)

    print(f"Train Loss: {train_loss:.4f}, Accuracy: {train_acc:.2f}%")
    print(f"Validation Loss: {val_loss:.4f}, Accuracy: {val_acc:.2f}%")
    print(f"Precision: {val_precision:.2f}, Recall: {val_recall:.2f}, F1-Score: {val_f1:.2f}")
    print(f"Confusion Matrix:\n{val_cm}")

    # Save Best Model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "best_model.pth")
        print(f"✅ New best model saved with validation accuracy: {val_acc:.2f}%")

    # Adjust Learning Rate
    lr_scheduler.step(val_acc)

# Plot Confusion Matrix
def plot_confusion_matrix(cm, class_names):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()

# Display Confusion Matrix
plot_confusion_matrix(val_cm, class_names=train_dataset.classes)