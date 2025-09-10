import os
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from efficientnet_pytorch import EfficientNet
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Paths
base_dir = 'D:\\Users\\Majid\\RES_NET\\New_Data\\dataset'
train_dir = os.path.join(base_dir, 'Train')
val_dir = os.path.join(base_dir, 'Validation')
test_dir = os.path.join(base_dir, 'Test')
model_path = "D:\\Users\\Majid\\RES_NET\\New_Networks\\EfficientNet_b7_March17.pth"

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transforms
transform = transforms.Compose([
    transforms.Resize((600, 600)),
    transforms.ToTensor(),
])

# Load EfficientNet-B7 model
model = EfficientNet.from_name('efficientnet-b7')
model._fc = torch.nn.Linear(model._fc.in_features, 2)  # Binary classification
state_dict = torch.load(model_path, map_location=device)
model.load_state_dict(state_dict)
model.to(device)
model.eval()

# Helper: get image names from a directory
def get_all_image_names(directory):
    image_list = []
    for label in ['bad_cut', 'good_cut']:
        subdir = os.path.join(directory, label)
        for fname in os.listdir(subdir):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_list.append(fname)
    return image_list

# Confusion matrix plot function
def plot_confusion_matrix(y_true, y_pred, class_names, normalize=False, title='Confusion Matrix'):
    cm = confusion_matrix(y_true, y_pred, normalize='true' if normalize else None)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues, values_format=".2f" if normalize else "d")
    plt.title(title)
    plt.show()

# Collect image names
train_images = get_all_image_names(train_dir)
val_images = get_all_image_names(val_dir)

# Load test dataset
test_dataset = ImageFolder(test_dir, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Predict
test_results = []
true_labels = []
pred_labels = []

for i, (inputs, labels) in enumerate(test_loader):
    inputs = inputs.to(device)
    labels = labels.to(device)
    with torch.no_grad():
        outputs = model(inputs)
        preds = torch.argmax(outputs, 1)

    # Get image name
    image_path, _ = test_loader.dataset.samples[i]
    image_name = os.path.basename(image_path)

    test_results.append({'image_name': image_name, 'predicted_label': preds.item()})
    true_labels.append(labels.item())
    pred_labels.append(preds.item())

# Pad lists to be same length
max_len = max(len(train_images), len(val_images), len(test_results))
pad = lambda l: l + [''] * (max_len - len(l))

# Save CSV
csv_data = {
    'train_images': pad(train_images),
    'val_images': pad(val_images),
    'test_image_name': pad([r['image_name'] for r in test_results]),
    'predicted_label': pad([r['predicted_label'] for r in test_results])
}

df = pd.DataFrame(csv_data)
df.to_csv('efficientnet_b7_results.csv', index=False)
print("✅ Results saved to efficientnet_b7_results.csv")

# Plot confusion matrix
plot_confusion_matrix(true_labels, pred_labels, class_names=['Bad Cut', 'Good Cut'], normalize=False)
