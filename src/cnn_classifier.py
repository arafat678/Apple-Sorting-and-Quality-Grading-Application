"""
cnn_classifier.py
=================
Deep Learning model definition, fine-tuning, and inference using PyTorch
and torchvision / FastAI for multi-class apple defect and quality grading.
"""

import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder


def build_apple_classifier(num_classes=3, pretrained=True):
    """
    Constructs a transfer-learning ResNet-50 classifier adapted for
    apple quality grading (Grade A, Grade B, Grade C).
    """
    weights = models.ResNet50_Weights.DEFAULT if pretrained else None
    model = models.resnet50(weights=weights)

    # Replace final classification head
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes)
    )
    return model


def get_data_transforms():
    """
    Standard data augmentation and normalization pipeline.
    """
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return train_transform, val_transform


def predict_quality_grade(model, image_path, class_names=None, device='cpu'):
    """
    Evaluates an apple image and returns predicted quality grade and probability distribution.
    """
    from PIL import Image

    if class_names is None:
        class_names = ["Grade_A_Premium", "Grade_B_Standard", "Grade_C_Defective"]

    _, val_transform = get_data_transforms()
    img = Image.open(image_path).convert('RGB')
    tensor = val_transform(img).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)[0].cpu().numpy()

    pred_idx = int(torch.argmax(outputs, dim=1).item())
    return {
        "predicted_grade": class_names[pred_idx],
        "confidence": float(probs[pred_idx]),
        "probabilities": {class_names[i]: float(probs[i]) for i in range(len(class_names))}
    }

