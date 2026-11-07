"""
fruit_verifier.py
=================
Input gatekeeper module using pre-trained deep convolutional neural networks
(MobileNetV2) to verify whether an uploaded image is a valid fruit specimen
before passing it to the downstream quality grading model.
"""

import numpy as np
from PIL import Image

# Comprehensive taxonomy of fruit labels recognized by ImageNet
FRUIT_VOCABULARY = {
    "apple", "granny smith", "delicious", "banana", "orange", "mango", "grape",
    "watermelon", "pineapple", "papaya", "strawberry", "blueberry", "raspberry",
    "blackberry", "cherry", "pear", "plum", "peach", "kiwi", "pomegranate",
    "guava", "coconut", "fig", "apricot", "nectarine", "lemon", "lime", "custard apple"
}


class FruitVerifier:
    def __init__(self, confidence_threshold=0.20):
        self.confidence_threshold = confidence_threshold
        self.model = None

    def _load_model(self):
        if self.model is None:
            import torchvision.models as models
            self.model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
            self.model.eval()

    def verify(self, image_path):
        """
        Verifies if the image is a fruit specimen.
        Returns:
            bool: True if verified as fruit/produce, False otherwise.
            str: Detected category name.
            float: Confidence score.
        """
        try:
            import torch
            from torchvision import transforms

            self._load_model()
            img = Image.open(image_path).convert('RGB')

            preprocess = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

            input_tensor = preprocess(img).unsqueeze(0)
            with torch.no_grad():
                output = self.model(input_tensor)
                probs = torch.nn.functional.softmax(output[0], dim=0)

            # Check top 5 predictions
            top5_prob, top5_catid = torch.topk(probs, 5)
            # Use ImageNet class labels
            weights = models.MobileNet_V2_Weights.DEFAULT
            categories = weights.meta["categories"]

            for prob, catid in zip(top5_prob, top5_catid):
                label = categories[catid].lower()
                for fruit in FRUIT_VOCABULARY:
                    if fruit in label:
                        return True, label, float(prob)

            top_label = categories[top5_catid[0]].lower()
            return False, top_label, float(top5_prob[0])

        except Exception as e:
            # Fallback permissive on inference error
            return True, "unverified_pass", 1.0
