"""
contour_analyzer.py
===================
Computer Vision module for physical trait extraction, geometric measurement,
and surface defect contour segmentation on apple specimens using OpenCV.
"""

import cv2
import numpy as np


def extract_physical_traits(image_path):
    """
    Extracts geometric dimensions (width, height, aspect ratio), convex hull,
    and defect surface area from an input apple image.

    Returns:
        dict: Containing width (px), height (px), aspect_ratio,
              contour_area, hull_area, solidity, and defect_ratio.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image at {image_path}")

    # Convert to grayscale & smooth
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Edge detection & thresholding
    edged = cv2.Canny(blurred, 50, 150)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Find external contours
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        # Fallback to threshold contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return {
            "width": None,
            "height": None,
            "aspect_ratio": None,
            "solidity": None,
            "defect_ratio": 0.0
        }

    # Select primary fruit contour by maximum area
    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    area = cv2.contourArea(c)
    hull = cv2.convexHull(c)
    hull_area = cv2.contourArea(hull)

    solidity = float(area) / hull_area if hull_area > 0 else 1.0
    aspect_ratio = float(w) / h if h > 0 else 1.0

    # Defect segmentation: deviation between Convex Hull and perimeter
    defect_area = max(0, hull_area - area)
    defect_ratio = defect_area / hull_area if hull_area > 0 else 0.0

    return {
        "width": int(w),
        "height": int(h),
        "aspect_ratio": round(aspect_ratio, 3),
        "contour_area": int(area),
        "hull_area": int(hull_area),
        "solidity": round(solidity, 3),
        "defect_ratio": round(defect_ratio, 4)
    }


def draw_traits_overlay(image_path, output_path=None):
    """
    Draws bounding box, contour perimeter, and dimensions directly onto the image.
    """
    img = cv2.imread(image_path)
    traits = extract_physical_traits(image_path)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.drawContours(img, [c], -1, (255, 0, 0), 2)

        label = f"W: {w}px, H: {h}px | AR: {traits['aspect_ratio']}"
        cv2.putText(img, label, (x, max(20, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    if output_path:
        cv2.imwrite(output_path, img)
    return img, traits

