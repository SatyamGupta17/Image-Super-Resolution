import os
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import cv2
import numpy as np
from keras.models import load_model, Model
from keras.layers import Input
from prediction import create_gen
from prediction1 import predict_original_size

# Flask app setup
app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Model loading
MODEL_PATH = "GAN/gen_e_20.h5"
generator_model = None

def load_generator_model():
    global generator_model
    dummy_input = Input(shape=(None, None, 3))
    generator_model = create_gen(dummy_input, num_res_block=16)
    generator_model.load_weights(MODEL_PATH)

# Load model at startup
load_generator_model()

@app.route('/')
def home():
    return render_template('index.html', lr_url=None, sr_url=None, comp_url=None)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(url_for('home'))

    file = request.files['file']
    if file.filename == '':
        return redirect(url_for('home'))

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    print(f"Prediction started for uploaded file: {filename}")
    try:
        # Read the image using OpenCV
        lr_img = cv2.imread(filepath)
        if lr_img is None:
            return "Could not read the image file", 400 
        lr_img = cv2.cvtColor(lr_img, cv2.COLOR_BGR2RGB)
         
        result = predict_original_size(filepath)  
        if not result or 'urls' not in result:
            return "Prediction failed", 500
         
        return render_template('index.html',
                            input_image=result['urls']['lr'],
                            output_image=result['urls']['sr'],
                            comparison_image=result['urls']['hr'])
    
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return f"An error occurred: {str(e)}", 500 

if __name__ == '__main__':
    app.run(debug=True)
