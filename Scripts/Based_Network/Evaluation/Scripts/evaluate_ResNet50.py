import os
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
from torchvision import transforms, datasets
from torchvision.models import resnet50
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Paths
data_dir = "D:\\Users\\Majid\\RES_NET\\New_Data\\dataset"
model_path = "D:\\Users\\Majid\\RES_NET\\New_Networks\\Resnet50_big.pth"

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transforms (match training)
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Load ResNet50 model
model = resnet50(weights=None)
model.fc = nn.Linear(model.fc.in_features, 2)  # Binary classification
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# Load datasets
train_dataset = datasets.ImageFolder(os.path.join(data_dir, "Train"), transform=transform)
val_dataset = datasets.ImageFolder(os.path.join(data_dir, "Validation"), transform=transform)
test_dataset = datasets.ImageFolder(os.path.join(data_dir, "Test"), transform=transform)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Get image names
def get_image_names(folder):
    image_names = []
    for class_dir in ['bad_cut', 'good_cut']:
        full_dir = os.path.join(folder, class_dir)
        image_names.extend(os.listdir(full_dir))
    return image_names

train_images = get_image_names(os.path.join(data_dir, 'Train'))
val_images = get_image_names(os.path.join(data_dir, 'Validation'))

# Run inference
true_labels = []
pred_labels = []
test_results = []

for idx, (inputs, labels) in enumerate(test_loader):
    inputs = inputs.to(device)
    labels = labels.to(device)

    with torch.no_grad():
        outputs = model(inputs)
        preds = torch.argmax(outputs, dim=1)

    image_path, _ = test_dataset.samples[idx]
    image_name = os.path.basename(image_path)

    test_results.append({
        "image_name": image_name,
        "predicted_label": preds.item()
    })

    true_labels.append(labels.item())
    pred_labels.append(preds.item())

# Save to CSV
max_len = max(len(train_images), len(val_images), len(test_results))
pad = lambda l: l + [''] * (max_len - len(l))

df = pd.DataFrame({
    "train_images": pad(train_images),
    "val_images": pad(val_images),
    "test_image_name": pad([r["image_name"] for r in test_results]),
    "predicted_label": pad([r["predicted_label"] for r in test_results]),
})
df.to_csv("resnet50_results.csv", index=False)
print("✅ Results saved to resnet50_results.csv")

# Plot confusion matrix
def plot_confusion(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix")
    plt.show()

plot_confusion(true_labels, pred_labels, class_names=['Bad Cut', 'Good Cut'])
