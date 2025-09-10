import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torch.optim as optim
import os
from torchvision import models
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.tensorboard import SummaryWriter

# Additional imports for Grad-CAM
import cv2
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

# Check GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Paths to dataset
data_dir = "D:\\Users\\Majid\\RES_NET\\New_Data\\dataset"
train_dir = os.path.join(data_dir, "Train")
valid_dir = os.path.join(data_dir, "Validation")

# Define Transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load Dataset
train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
valid_dataset = datasets.ImageFolder(root=valid_dir, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False, num_workers=0)

# Load MobileNetV3 Small Model with pretrained weights
model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
# Modify the classifier's last layer for binary classification (2 classes)
model.classifier[3] = nn.Linear(in_features=1024, out_features=2)
model = model.to(device)

# Define Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Initialize TensorBoard writer
writer = SummaryWriter()

# Early Stopping Class
class EarlyStopping:
    def __init__(self, patience=5, delta=0.001):
        """
        Args:
            patience (int): How many epochs to wait after last time validation loss improved.
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
        """
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None or val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

# Instantiate early stopping
early_stopping = EarlyStopping(patience=10, delta=0.001)

# Class names for labeling
class_names = ["Good Cut", "Bad Cut"]

# Training Loop Settings
num_epochs = 70  # Adjust if needed based on your dataset performance
best_val_acc = 0.0

train_acc_list = []
val_acc_list = []
train_f1_list = []
val_f1_list = []
train_loss_list = []
val_loss_list = []
train_confidence_list = []
val_confidence_list = []

# Define softmax for confidence computation
softmax = nn.Softmax(dim=1)

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    y_true_train, y_pred_train = [], []
    train_confidences = []

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        probabilities = softmax(outputs)
        max_confidences, _ = probabilities.max(dim=1)
        train_confidences.extend(max_confidences.detach().cpu().numpy())
        
        _, predicted = outputs.max(1)
        y_true_train.extend(labels.cpu().numpy())
        y_pred_train.extend(predicted.cpu().numpy())

    # Calculate Train Metrics
    train_acc = accuracy_score(y_true_train, y_pred_train) * 100
    train_f1 = f1_score(y_true_train, y_pred_train, average='weighted') * 100
    avg_train_confidence = np.mean(train_confidences) * 100
    train_loss = running_loss / len(train_loader)
    train_loss_list.append(train_loss)
    train_acc_list.append(train_acc)
    train_f1_list.append(train_f1)
    train_confidence_list.append(avg_train_confidence)

    # Log training metrics to TensorBoard
    writer.add_scalar('Train/Loss', train_loss, epoch)
    writer.add_scalar('Train/Accuracy', train_acc, epoch)
    writer.add_scalar('Train/F1-score', train_f1, epoch)
    writer.add_scalar('Train/Confidence', avg_train_confidence, epoch)

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
          f"Train F1: {train_f1:.2f}%, Train Confidence: {avg_train_confidence:.2f}%")

    # Validation Step
    model.eval()
    y_true_val, y_pred_val = [], []
    val_confidences = []
    running_val_loss = 0.0

    with torch.no_grad():
        for images, labels in valid_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_val_loss += loss.item()
            probabilities = softmax(outputs)
            max_confidences, _ = probabilities.max(dim=1)
            val_confidences.extend(max_confidences.detach().cpu().numpy())
            
            _, predicted = outputs.max(1)
            y_true_val.extend(labels.cpu().numpy())
            y_pred_val.extend(predicted.cpu().numpy())

    val_loss_epoch = running_val_loss / len(valid_loader)
    val_acc = accuracy_score(y_true_val, y_pred_val) * 100
    val_f1 = f1_score(y_true_val, y_pred_val, average='weighted') * 100
    avg_val_confidence = np.mean(val_confidences) * 100

    val_loss_list.append(val_loss_epoch)
    val_acc_list.append(val_acc)
    val_f1_list.append(val_f1)
    val_confidence_list.append(avg_val_confidence)

    # Log validation metrics to TensorBoard
    writer.add_scalar('Validation/Loss', val_loss_epoch, epoch)
    writer.add_scalar('Validation/Accuracy', val_acc, epoch)
    writer.add_scalar('Validation/F1-score', val_f1, epoch)
    writer.add_scalar('Validation/Confidence', avg_val_confidence, epoch)

    print(f"Validation Acc: {val_acc:.2f}%, Validation F1: {val_f1:.2f}%, "
          f"Validation Confidence: {avg_val_confidence:.2f}%")

    # Save Best Model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "mobilenet_v3_small_best.pth")
        print("Saved Best Model!")

    # Print Precision & Recall for the epoch
    precision = precision_score(y_true_val, y_pred_val, average='weighted') * 100
    recall = recall_score(y_true_val, y_pred_val, average='weighted') * 100
    print(f"Precision: {precision:.2f}%, Recall: {recall:.2f}%")

    # Early Stopping Check
    early_stopping(val_loss_epoch)
    if early_stopping.early_stop:
        print("Early stopping triggered. Stopping training.")
        break

# Plot Accuracy & F1-score per Epoch
plt.figure(figsize=(10, 5))
plt.plot(range(1, len(train_acc_list)+1), train_acc_list, label='Train Accuracy', marker='o')
plt.plot(range(1, len(val_acc_list)+1), val_acc_list, label='Validation Accuracy', marker='o')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Train & Validation Accuracy')
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(range(1, len(train_f1_list)+1), train_f1_list, label='Train F1-score', marker='o')
plt.plot(range(1, len(val_f1_list)+1), val_f1_list, label='Validation F1-score', marker='o')
plt.xlabel('Epoch')
plt.ylabel('F1-score (%)')
plt.title('Train & Validation F1-score')
plt.legend()
plt.grid()
plt.show()

# Plot Loss per Epoch
plt.figure(figsize=(10, 5))
plt.plot(range(1, len(train_loss_list)+1), train_loss_list, label='Train Loss', marker='o')
plt.plot(range(1, len(val_loss_list)+1), val_loss_list, label='Validation Loss', marker='o')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Train & Validation Loss')
plt.legend()
plt.grid()
plt.show()

# Plot Confidence Scores per Epoch
plt.figure(figsize=(10, 5))
plt.plot(range(1, len(train_confidence_list)+1), train_confidence_list, label='Train Confidence', marker='o')
plt.plot(range(1, len(val_confidence_list)+1), val_confidence_list, label='Validation Confidence', marker='o')
plt.xlabel('Epoch')
plt.ylabel('Confidence (%)')
plt.title('Train & Validation Confidence Scores')
plt.legend()
plt.grid()
plt.show()

# Confusion Matrix for Validation using class names
cm = confusion_matrix(y_true_val, y_pred_val)
plt.figure(figsize=(6,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()

# Close TensorBoard writer
writer.close()

print("Training Completed!")

# ---------------------------
# Grad-CAM Integration Section
# ---------------------------
def generate_gradcam(model, input_tensor, target_layer, class_idx=None):
    """
    Generates and visualizes Grad-CAM for the given input_tensor.
    Args:
        model: The trained model.
        input_tensor: Preprocessed image tensor of shape (1, C, H, W).
        target_layer: The convolutional layer to target for Grad-CAM.
        class_idx: The index of the target class; if None, the predicted class is used.
    """
    model.eval()
    # Initialize GradCAM with the model and target layer
    cam = GradCAM(model=model, target_layers=[target_layer], use_cuda=torch.cuda.is_available())
    grayscale_cam = cam(input_tensor=input_tensor, target_category=class_idx)
    grayscale_cam = grayscale_cam[0, :]
    
    # Prepare the original image for visualization
    img = input_tensor.squeeze().cpu().numpy().transpose(1, 2, 0)
    # Reverse normalization (using ImageNet means and stds)
    img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
    img = np.clip(img, 0, 1)
    
    # Overlay the heatmap on the image
    cam_image = show_cam_on_image(img, grayscale_cam, use_rgb=True)
    
    plt.figure(figsize=(8, 8))
    plt.imshow(cam_image)
    plt.title("Grad-CAM")
    plt.axis("off")
    plt.show()

# Example usage of Grad-CAM on a sample validation image
# (Select an image from the validation dataset; here we use the first image)
sample_img, sample_label = valid_dataset[0]
input_tensor = sample_img.unsqueeze(0).to(device)
# For MobileNetV3 Small, we can choose the last feature layer as the target
target_layer = model.features[-1]
# Optionally, specify a target class index (or leave as None to use the predicted class)
generate_gradcam(model, input_tensor, target_layer, class_idx=None)
