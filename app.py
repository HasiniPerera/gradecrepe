from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import os
from PIL import Image
import tensorflow as tf
import numpy as np

app = Flask(__name__)

# This is the path to the upload directory
app.config['UPLOAD_FOLDER'] = 'uploads/'
# Allowed extensions for file upload
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Make sure the upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Function to check allowed file types
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Load the TensorFlow model

model = tf.keras.models.load_model('D:\IIT\Project\Rubber_Grade_last.h5')
CLASS_LABELS = ['Grade 1', 'Grade 1X', 'Grade 2', 'Grade 3', 'Grade 4']

# Define the prediction function
def predict(image_path, model):
    img = Image.open(image_path)
    img = img.resize((224, 224))
    img = np.array(img)
    img = img.astype(np.float32)
    img = np.expand_dims(img, axis=0)
    img /= 255.0

    predictions = model.predict(img)
    score = tf.nn.softmax(predictions[0])

    # Determine the class with the highest confidence
    class_id = np.argmax(score)
    class_label = CLASS_LABELS[class_id]
    confidence = np.max(score)

    return class_label, confidence

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        # If user does not select file, browser also submits an empty part without filename
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # Make prediction
            
            class_label, confidence = predict(file_path, model)
            print(class_label, confidence)
            

            return render_template('result.html',class_label=class_label, confidence="{:.2f}%".format(confidence * 100))
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
