from flask import Flask, render_template, request
import torch
import pickle
import librosa
import numpy as np


model = pickle.load(open("./Models/model(L1).txt", "rb"))

app = Flask(__name__)

# App autoreloads
app.config["TEMPLATES_AUTO_RELOAD"] = True

@app.after_request
def after_request(response):
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Expires"] = 0
    response.headers["Pragma"] = "no-cache"
    return response

@app.route("/")
def index():
    """Home page"""
    return render_template('homepage.html')

@app.route("/diagnosis", methods = ['GET', 'POST'])
def diagnose():
    """Take file, process, return diagnosis"""
    if request.method == 'POST':
        file = request.files['file']
        # fileName = secure_filename(file.filename)
        # file.save(os.path.join("./FlaskUploads/", fileName))

        audio, sampleRate = librosa.load(file)
        mfccs = librosa.feature.mfcc(y=audio, sr=sampleRate, n_mfcc=40)

        expectedSize = 51600

        soundArray = mfccs.flatten()
        soundArray = np.pad(soundArray, (0, expectedSize - soundArray.shape[0]), 'constant')

        prediction = model(torch.tensor(soundArray))

        negativeValue = prediction[1] * 100

        return render_template('diagnosis.html', val=round(negativeValue.item(), 2))
    else:
        return render_template('homepage.html')