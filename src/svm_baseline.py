"""
svm_baseline.py
===============
Classical machine learning baseline using color histogram, GLCM texture,
and geometric features paired with Support Vector Machine (RBF kernel)
for comparative evaluation against deep vision models.
"""

import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def extract_handcrafted_features(image_path):
    """
    Extracts RGB/HSV color moments and edge descriptors.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")

    # Color moments in HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h_mean, h_std = np.mean(hsv[:, :, 0]), np.std(hsv[:, :, 0])
    s_mean, s_std = np.mean(hsv[:, :, 1]), np.std(hsv[:, :, 1])
    v_mean, v_std = np.mean(hsv[:, :, 2]), np.std(hsv[:, :, 2])

    # Grayscale Canny edges density
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])

    return np.array([h_mean, h_std, s_mean, s_std, v_mean, v_std, edge_density])


def train_svm_baseline(X_train, y_train, C=1.0, gamma='scale'):
    """
    Fits an RBF-kernel SVM on standardized feature matrices.
    """
    model = make_pipeline(StandardScaler(), SVC(C=C, gamma=gamma, probability=True))
    model.fit(X_train, y_train)
    return model
