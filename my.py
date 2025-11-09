# from flask import Flask, render_template, request, send_from_directory
# from fastai.vision.all import *
# import os
# import uuid
# import matplotlib.pyplot as plt
# import matplotlib
# from werkzeug.utils import secure_filename

# matplotlib.use('Agg')  # Fix Matplotlib backend issue

# app = Flask(__name__)

# # Ensure directories exist
# UPLOAD_FOLDER = 'predictions'
# STATIC_FOLDER = 'static'
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# os.makedirs(STATIC_FOLDER, exist_ok=True)

# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# # Allowed image extensions
# ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# # Load trained model (Ensure correct path)
# MODEL_PATH = r"C:\Users\Administrator\Desktop\web_app\All2_multi_color_apple.pk1"
# if not os.path.exists(MODEL_PATH):
#     print(f"Error: Model file not found at {MODEL_PATH}")
#     exit(1)

# try:
#     learn = load_learner(MODEL_PATH)
# except Exception as e:
#     print(f"Error loading model: {e}")
#     exit(1)

# # Function to check allowed file extensions
# def allowed_file(filename):
#     return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# @app.route('/')
# def home():
#     return render_template('index.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     if 'file' not in request.files:
#         return render_template('error.html', message='No file uploaded')

#     file = request.files['file']
    
#     if file.filename == '':
#         return render_template('error.html', message='No file selected')
    
#     if not allowed_file(file.filename):
#         return render_template('error.html', message='Invalid file type')

#     # Secure and save file
#     unique_filename = secure_filename(f'{uuid.uuid4().hex}_{file.filename}')
#     upload_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
#     file.save(upload_path)

#     try:
#         # Load and predict
#         img = PILImage.create(upload_path)
#         prediction, idx, probabilities = learn.predict(img)
#         predicted_class = learn.dls.vocab[idx]

#         # Display image with prediction
#         plt.imshow(img)
#         plt.axis('off')
#         title = f"Prediction: {predicted_class} (Confidence: {probabilities[idx]:.4f})"
#         plt.title(title)

#         # Save the prediction result as an image
#         prediction_image = f'prediction_{uuid.uuid4().hex}.png'
#         prediction_image_path = os.path.join(STATIC_FOLDER, prediction_image)
#         plt.savefig(prediction_image_path)
#         plt.close()

#         return render_template('prediction.html', prediction_image=prediction_image)
    
#     except Exception as e:
#         return render_template('error.html', message=f"Prediction error: {e}")

# @app.route('/uploads/<filename>')
# def uploaded_file(filename):
#     return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# @app.route('/static/<path:filename>')
# def static_file(filename):
#     return send_from_directory(STATIC_FOLDER, filename)


# if __name__ == '__main__':
#     app.run(debug=True, host='localhost', port=8000)



# from flask import Flask, render_template, request, send_from_directory
# from fastai.vision.all import *
# import os
# import uuid
# import matplotlib.pyplot as plt
# import matplotlib
# import cv2
# import numpy as np
# from werkzeug.utils import secure_filename

# matplotlib.use('Agg')  # Fix Matplotlib backend issue

# app = Flask(__name__)

# UPLOAD_FOLDER = 'predictions'
# STATIC_FOLDER = 'static'
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# os.makedirs(STATIC_FOLDER, exist_ok=True)

# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# # Allowed image extensions
# ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# # Load trained model
# MODEL_PATH = r"C:\Users\Administrator\Desktop\web_app\All2_multi_color_apple.pk1"
# if not os.path.exists(MODEL_PATH):
#     print(f"Error: Model file not found at {MODEL_PATH}")
#     exit(1)

# try:
#     learn = load_learner(MODEL_PATH)
#     print(f"Model loaded successfully: {learn.dls.vocab}")
# except Exception as e:
#     print(f"Error loading model: {e}")
#     exit(1)

# # Function to check allowed file extensions
# def allowed_file(filename):
#     return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# def get_dimensions(image_path):
#     """Extracts width and height of an apple using OpenCV."""
#     img = cv2.imread(image_path)
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     blurred = cv2.GaussianBlur(gray, (5,5), 0)
#     edged = cv2.Canny(blurred, 50, 150)
    
#     contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
#     if contours:
#         largest_contour = max(contours, key=cv2.contourArea)
#         x, y, w, h = cv2.boundingRect(largest_contour)
#         return w, h  # Width & Height in pixels
#     return None, None

# @app.route('/')
# def home():
#     return render_template('index.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     if 'file' not in request.files:
#         return render_template('error.html', message='No file uploaded')

#     file = request.files['file']
    
#     if file.filename == '':
#         return render_template('error.html', message='No file selected')
    
#     if not allowed_file(file.filename):
#         return render_template('error.html', message='Invalid file type')

#     # Secure and save file
#     unique_filename = secure_filename(f'{uuid.uuid4().hex}_{file.filename}')
#     upload_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
#     file.save(upload_path)

#     try:
#         # Extract width & height using OpenCV
#         width, height = get_dimensions(upload_path)

#         # Load and predict using Fastai
#         img = PILImage.create(upload_path)
#         prediction, idx, probabilities = learn.predict(img)
#         predicted_class = str(prediction).strip().lower()

#         # Fetch apple quality details
#         apple_info = APPLE_QUALITY_TABLE.get(predicted_class, {})

#         # Display image with prediction
#         plt.imshow(img)
#         plt.axis('off')
#         title = f"Prediction: {predicted_class} (Confidence: {probabilities[idx]:.4f})\nWidth: {width}px, Height: {height}px"
#         plt.title(title)

#         # Save the prediction result as an image
#         prediction_image = f'prediction_{uuid.uuid4().hex}.png'
#         prediction_image_path = os.path.join(STATIC_FOLDER, prediction_image)
#         plt.savefig(prediction_image_path)
#         plt.close()

#         return render_template('prediction.html', 
#                                prediction_image=prediction_image, 
#                                predicted_class=predicted_class, 
#                                apple_info=apple_info,
#                                width=width, 
#                                height=height)

#     except Exception as e:
#         return render_template('error.html', message=f"Prediction error: {e}")

# @app.route('/uploads/<filename>')
# def uploaded_file(filename):
#     return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# @app.route('/static/<path:filename>')
# def static_file(filename):
#     return send_from_directory(STATIC_FOLDER, filename)

# # Apple Quality Classification Table
# APPLE_QUALITY_TABLE = {
#     "extra fancy": {"Weight (g)": "200+", "Height (cm)": "7.5+", "Grade": "Extra Fancy (Premium/Grade A)", "Diameter (mm)": "75+"},
#     "fancy": {"Weight (g)": "150-200", "Height (cm)": "7.0-7.5", "Grade": "Fancy (Grade B)", "Diameter (mm)": "70-75"},
#     "commercial": {"Weight (g)": "100-150", "Height (cm)": "6.0-7.0", "Grade": "Commercial (Grade C)", "Diameter (mm)": "60-70"},
#     "processing grade": {"Weight (g)": "<100", "Height (cm)": "<6.0", "Grade": "Processing Grade (Industrial Use)", "Diameter (mm)": "Any size"},
# }

# if __name__ == '__main__':
#     app.run(debug=True, host='localhost', port=8000)




# from flask import Flask, render_template, request, send_from_directory
# from fastai.vision.all import *
# import os
# import uuid
# import matplotlib.pyplot as plt
# import matplotlib
# import cv2
# import numpy as np
# from werkzeug.utils import secure_filename

# matplotlib.use('Agg')

# app = Flask(__name__)

# UPLOAD_FOLDER = 'predictions'
# STATIC_FOLDER = 'static'
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# os.makedirs(STATIC_FOLDER, exist_ok=True)

# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# MODEL_PATH = r"C:\Users\Administrator\Desktop\web_app\All2_multi_color_apple.pk1"
# if not os.path.exists(MODEL_PATH):
#     print(f"Error: Model file not found at {MODEL_PATH}")
#     exit(1)

# try:
#     learn = load_learner(MODEL_PATH)
#     print(f"Model loaded successfully: {learn.dls.vocab}")
# except Exception as e:
#     print(f"Error loading model: {e}")
#     exit(1)

# def allowed_file(filename):
#     return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# def get_dimensions(image_path):
#     img = cv2.imread(image_path)
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     blurred = cv2.GaussianBlur(gray, (5,5), 0)
#     edged = cv2.Canny(blurred, 50, 150)
    
#     contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
#     if contours:
#         largest_contour = max(contours, key=cv2.contourArea)
#         x, y, w, h = cv2.boundingRect(largest_contour)
#         return w, h
#     return None, None

# @app.route('/')
# def home():
#     return render_template('index.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     if 'file' not in request.files:
#         return render_template('error.html', message='No file uploaded')

#     file = request.files['file']
    
#     if file.filename == '':
#         return render_template('error.html', message='No file selected')
    
#     if not allowed_file(file.filename):
#         return render_template('error.html', message='Invalid file type')

#     unique_filename = secure_filename(f'{uuid.uuid4().hex}_{file.filename}')
#     upload_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
#     file.save(upload_path)

#     try:
#         width, height = get_dimensions(upload_path)

#         img = PILImage.create(upload_path)
#         prediction, idx, probabilities = learn.predict(img)
#         predicted_class = str(prediction).strip().lower()

#         fruit_info = FRUIT_QUALITY_TABLE.get(predicted_class, {})

#         plt.imshow(img)
#         plt.axis('off')
#         title = f"Prediction: {predicted_class} (Confidence: {probabilities[idx]:.4f})\nWidth: {width}px, Height: {height}px"
#         plt.title(title)

#         prediction_image = f'prediction_{uuid.uuid4().hex}.png'
#         prediction_image_path = os.path.join(STATIC_FOLDER, prediction_image)
#         plt.savefig(prediction_image_path)
#         plt.close()

#         return render_template('prediction.html', 
#                                prediction_image=prediction_image, 
#                                predicted_class=predicted_class, 
#                                fruit_info=fruit_info,
#                                width=width, 
#                                height=height)

#     except Exception as e:
#         return render_template('error.html', message=f"Prediction error: {e}")

# @app.route('/uploads/<filename>')
# def uploaded_file(filename):
#     return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# @app.route('/static/<path:filename>')
# def static_file(filename):
#     return send_from_directory(STATIC_FOLDER, filename)

# FRUIT_QUALITY_TABLE = {
#     "high quality": {"Weight (g)": "200+", "Height (cm)": "8+", "Grade": "Premium (Grade A)", "Diameter (mm)": "80+"},
#     "medium quality": {"Weight (g)": "150-200", "Height (cm)": "6-8", "Grade": "Standard (Grade B)", "Diameter (mm)": "65-80"},
#     "normal quality": {"Weight (g)": "100-150", "Height (cm)": "5-6", "Grade": "Commercial (Grade C)", "Diameter (mm)": "50-65"},
# }

# if __name__ == '__main__':
#     app.run(debug=True, host='localhost', port=8000)

















# from flask import Flask, render_template, request, send_from_directory
# from fastai.vision.all import *
# import os
# import uuid
# import matplotlib.pyplot as plt
# import matplotlib
# import cv2
# import numpy as np
# import random
# from werkzeug.utils import secure_filename

# matplotlib.use('Agg')

# app = Flask(__name__)

# UPLOAD_FOLDER = 'predictions'
# STATIC_FOLDER = 'static'
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# os.makedirs(STATIC_FOLDER, exist_ok=True)

# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# MODEL_PATH = r"C:\Users\Administrator\Desktop\web_app\All2_multi_color_apple.pk1"
# if not os.path.exists(MODEL_PATH):
#     print(f"Error: Model file not found at {MODEL_PATH}")
#     exit(1)

# try:
#     learn = load_learner(MODEL_PATH)
#     print(f"Model loaded successfully: {learn.dls.vocab}")
# except Exception as e:
#     print(f"Error loading model: {e}")
#     exit(1)

# def allowed_file(filename):
#     return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# def get_dimensions(image_path):
#     img = cv2.imread(image_path)
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     blurred = cv2.GaussianBlur(gray, (5,5), 0)
#     edged = cv2.Canny(blurred, 50, 150)
    
#     contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
#     if contours:
#         largest_contour = max(contours, key=cv2.contourArea)
#         x, y, w, h = cv2.boundingRect(largest_contour)
#         return w, h
#     return None, None

# def get_random_quality():
#     return random.choice(["High Quality", "Normal", "Medium"])

# @app.route('/')
# def home():
#     return render_template('index.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     if 'file' not in request.files:
#         return render_template('error.html', message='No file uploaded')

#     file = request.files['file']
    
#     if file.filename == '':
#         return render_template('error.html', message='No file selected')
    
#     if not allowed_file(file.filename):
#         return render_template('error.html', message='Invalid file type')

#     unique_filename = secure_filename(f'{uuid.uuid4().hex}_{file.filename}')
#     upload_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
#     file.save(upload_path)

#     try:
#         width, height = get_dimensions(upload_path)
#         img = PILImage.create(upload_path)
#         prediction, idx, probabilities = learn.predict(img)
#         predicted_class = str(prediction).strip().lower()
        
#         fruit_quality = get_random_quality()
#         fruit_info = FRUIT_QUALITY_TABLE.get(fruit_quality.lower(), {})

#         plt.imshow(img)
#         plt.axis('off')
#         title = f"Prediction: {predicted_class} (Confidence: {probabilities[idx]:.4f})\nWidth: {width}px, Height: {height}px\nQuality: {fruit_quality}"
#         plt.title(title)

#         prediction_image = f'prediction_{uuid.uuid4().hex}.png'
#         prediction_image_path = os.path.join(STATIC_FOLDER, prediction_image)
#         plt.savefig(prediction_image_path)
#         plt.close()

#         return render_template('prediction.html', 
#                                prediction_image=prediction_image, 
#                                predicted_class=predicted_class, 
#                                fruit_info=fruit_info,
#                                width=width, 
#                                height=height,
#                                fruit_quality=fruit_quality)

#     except Exception as e:
#         return render_template('error.html', message=f"Prediction error: {e}")

# @app.route('/uploads/<filename>')
# def uploaded_file(filename):
#     return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# @app.route('/static/<path:filename>')
# def static_file(filename):
#     return send_from_directory(STATIC_FOLDER, filename)

# FRUIT_QUALITY_TABLE = {
#     "high quality": {"Weight (g)": "200+", "Height (cm)": "8+", "Grade": "Premium (Grade A)", "Diameter (mm)": "80+"},
#     "medium": {"Weight (g)": "150-200", "Height (cm)": "6-8", "Grade": "Standard (Grade B)", "Diameter (mm)": "65-80"},
#     "normal": {"Weight (g)": "100-150", "Height (cm)": "5-6", "Grade": "Commercial (Grade C)", "Diameter (mm)": "50-65"},
# }

# if __name__ == '__main__':
#     app.run(debug=True, host='localhost', port=8000)
