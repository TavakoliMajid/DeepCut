import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import os
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import copy

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Data directories
data_dir = "D:\\Users\\Majid\\RES_NET\\New_Data\\dataset"
train_dir = os.path.join(data_dir, "Train")
val_dir = os.path.join(data_dir, "Validation")
test_dir = os.path.join(data_dir, "Test")

# Data transformations
transform = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# Load datasets
datasets_dict = {
    'train': datasets.ImageFolder(train_dir, transform=transform['train']),
    'val': datasets.ImageFolder(val_dir, transform=transform['val'])
}

dataloaders = {
    'train': DataLoader(datasets_dict['train'], batch_size=32, shuffle=True, num_workers=4),
    'val': DataLoader(datasets_dict['val'], batch_size=32, shuffle=False, num_workers=4)
}

# Class names
class_names = ['good_cut', 'bad_cut']

# Load ResNet-50 model and modify it for binary classification
model = models.resnet50(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(num_ftrs, 2)
)
model = model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.1, verbose=True)


def train_model(model, criterion, optimizer, scheduler, num_epochs=50, early_stop_patience=5):
    best_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    patience_counter = 0

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        print("-" * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(datasets_dict[phase])
            epoch_acc = running_corrects.double() / len(datasets_dict[phase])

            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            if phase == 'val':
                scheduler.step(epoch_loss)
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= early_stop_patience:
                    print("Early stopping triggered!")
                    model.load_state_dict(best_model_wts)
                    return model

    model.load_state_dict(best_model_wts)
    print(f"Best validation accuracy: {best_acc:.4f}")
    return model


if __name__ == '__main__':
    # Train the model
    model = train_model(model, criterion, optimizer, scheduler, num_epochs=50, early_stop_patience=5)

    # Save the trained model
    torch.save(model.state_dict(), "Resnet50_big.pth")

    # Load test dataset and evaluate
    model.eval()
    test_dataset = datasets.ImageFolder(test_dir, transform=transform['val'])
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

    y_true = []
    y_pred = []
    misclassified_images = []
    test_corrects = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            test_corrects += torch.sum(preds == labels.data)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

            for i in range(len(labels)):
                if labels[i] != preds[i]:
                    misclassified_images.append(test_dataset.imgs[i][0])

    test_acc = test_corrects.double() / len(test_dataset)
    print(f"Test Accuracy: {test_acc:.4f}")

    # Generate confusion matrix and classification report
    conf_matrix = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    print(conf_matrix)
    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))

    # Save results to CSV
    results_df = pd.DataFrame({
        "Image Path": [img[0] for img in test_dataset.imgs],
        "True Label": [class_names[label] for label in y_true],
        "Predicted Label": [class_names[pred] for pred in y_pred]
    })
    results_df.to_csv("classification_results_resnetBig.csv", index=False)

    misclassified_df = pd.DataFrame({
        "Misclassified Image Path": misclassified_images
    })
    misclassified_df.to_csv("misclassified_images_resnetBig.csv", index=False)

    print("Results saved to classification_results.csv and misclassified_images.csv")
