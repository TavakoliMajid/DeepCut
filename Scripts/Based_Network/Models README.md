# 🧠 Deep Learning Model Training Scripts

This repository contains Python training scripts for three convolutional neural network architectures, each fine-tuned for a binary image classification task (`Good Cut` vs `Bad Cut`):

- ✅ EfficientNet B7
- 📱 MobileNet V3 Small
- 🧱 ResNet50

---

## 📂 Files

| File Name                | Model               | Description                              |
|--------------------------|---------------------|------------------------------------------|
| `EFIICIENT_B7_2025.py`   | EfficientNet B7     | Trains and evaluates the EfficientNet B7 model. |
| `MobileNet_V3_Small.py`  | MobileNet V3 Small  | Trains and evaluates MobileNet V3 Small. |
| `ResNet50.py`            | ResNet50            | Trains and evaluates the ResNet50 model. |

---

## 📌 Requirements

Install required libraries:

```bash
pip install torch torchvision pandas numpy matplotlib scikit-learn
```

---

## 🏗 Project Structure

Each script includes:

- Data loading and augmentation
- Training and validation loops
- Accuracy and loss tracking
- Model checkpointing
- CSV export of training/validation metrics
- Confusion matrix and classification report
- Misclassified image logging

---

## 🧪 Dataset Structure

Expected input format:

```
dataset/
├── train/
│   ├── Good Cut/
│   └── Bad Cut/
├── val/
│   ├── Good Cut/
│   └── Bad Cut/
└── test/
    ├── Good Cut/
    └── Bad Cut/
```

---

## 🚀 Usage

To run each model:

```bash
python EFIICIENT_B7_2025.py
python MobileNet_V3_Small.py
python ResNet50.py
```

Each script saves:
- Trained model weights
- Accuracy/loss plots
- CSV with TP, FP, TN, FN, and accuracy
- Misclassified image names

---

## 📊 Output Example

Each script will generate:

- `model_name_results.csv` – includes:
  - True/Predicted labels
  - TP, FP, TN, FN counts
  - Accuracy, Precision, Recall, F1 Score
- `model_name.pth` – trained weights
- Visualization of training/validation loss and accuracy

---

## 📧 Author

Majid Tavakoli