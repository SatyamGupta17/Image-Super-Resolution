import os
import os.path as osp
import cv2
import numpy as np
import torch
import RRDBNet_arch as arch
from flask import Flask, render_template, request, send_from_directory, redirect, url_for

app = Flask(__name__)

# Ensure directories exist
os.makedirs("uploads", exist_ok=True)
os.makedirs("results", exist_ok=True)

# Load the model
model_path = 'models/RRDB_ESRGAN_x4.pth'  # Ensure the model file exists
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = arch.RRDBNet(3, 3, 64, 23, gc=32)
model.load_state_dict(torch.load(model_path, map_location=device), strict=True)
model.eval()
model = model.to(device)

@app.route('/')
def home():
    return render_template('index.html', input_image=None, output_image=None)  # Both images initially None

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(url_for('home'))

    file = request.files['file']
    if file.filename == '':
        return redirect(url_for('home'))

    filename = file.filename
    input_path = os.path.join("uploads", filename)
    file.save(input_path)  # Save original image

    # Read and process image
    img = cv2.imread(input_path, cv2.IMREAD_COLOR)
    img = img * 1.0 / 255
    img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
    img_LR = img.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()

    output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
    output = (output * 255.0).round().astype(np.uint8)

    output_filename = f"{osp.splitext(filename)[0]}_result.png"
    output_path = os.path.join("results", output_filename)
    cv2.imwrite(output_path, output)  # Save processed image

    return render_template(
        'index.html',
        input_image=url_for('get_upload', filename=filename),
        output_image=url_for('get_result', filename=output_filename)
    )

@app.route('/uploads/<filename>')
def get_upload(filename):
    return send_from_directory("uploads", filename)

@app.route('/results/<filename>')
def get_result(filename):
    return send_from_directory("results", filename)

if __name__ == "__main__":
    app.run(debug=True, port=5000)
