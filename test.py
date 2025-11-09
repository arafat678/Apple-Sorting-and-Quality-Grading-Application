

from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

# Load the pre-trained MobileNetV2 model
model = tf.keras.applications.MobileNetV2(weights='imagenet')

# Function to check if an image is a fruit
def is_fruit(image):
    image = image.resize((224, 224))  # Resize to match model input size
    img_array = np.array(image)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    predictions = model.predict(img_array)
    decoded_predictions = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=3)

    # List of fruit labels
    fruit_labels = [
        "apple", "banana", "orange", "mango", "grape", "watermelon", "pineapple", "papaya", "strawberry", "blueberry",
        "raspberry", "blackberry", "cherry", "pear", "plum", "peach", "kiwi", "pomegranate", "guava", "coconut",
        "dragon fruit", "durian", "rambutan", "lychee", "mangosteen", "passion fruit", "jackfruit", "starfruit",
        "longan", "snake fruit", "lemon", "lime", "tangerine", "grapefruit", "yuzu", "pomelo", "honeydew melon",
        "cantaloupe", "casaba melon", "gac fruit", "cranberry", "boysenberry", "gooseberry", "elderberry", "cloudberry",
        "mulberry", "loganberry", "apricot", "nectarine", "chikoo", "date", "jujube", "ackee", "buddha’s hand",
        "cherimoya", "feijoa", "loquat", "medlar", "noni", "santol", "miracle fruit", "ice cream bean",
        "horned melon", "jabuticaba", "cupuaçu"
    ]

    # Check if any of the top 3 predictions match fruit labels
    for _, label, _ in decoded_predictions[0]:
        if label.lower() in fruit_labels:
            return True
    return False

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    try:
        image = Image.open(io.BytesIO(file.read()))
        if is_fruit(image):
            return jsonify({'message': 'Fruit image detected'}), 200
        else:
            return jsonify({'message': 'It is not a fruit image'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    @app.route('/upload', methods=['POST'])
    def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    try:
        image = Image.open(io.BytesIO(file.read()))
        if is_fruit(image):
            return jsonify({'message': 'Fruit image detected'}), 200
        else:
            return jsonify({'message': 'I am sorry, I cannot recognize it. My developer is working on it.'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)

