import os
from urllib.request import HTTPErrorProcessor
# import pyrebase
from tensorflow.keras.utils import img_to_array
from keras.models import load_model
import cv2
import numpy as np
from werkzeug.exceptions import HTTPException

# config firebase line 11 - 24
# firebaseConfig = {
#   "apiKey": "",
#   "authDomain": "",
#   "projectId": "",
#   "storageBucket": "",
#   "messagingSenderId": "",
#   "appId": "",
#   "measurementId": "",
#    "databaseURL": "",
# };

# firebase = pyrebase.initialize_app(firebaseConfig)

# storage = firebase.storage()

from flask import *
from flask_restful import Resource, Api, reqparse

# edit class here
names = ['Boiled_leaves', 'Green_stalk_GradeA', 'Green_stalk_GradeB', 'Green_stalk_GradeC',
         'JUMBO', 'Red_stalk_GradeA', 'Red_stalk_GradeB', 'Red_stalk_GradeC']

app = Flask(__name__)
api = Api(app)

# Guide for uploading images to firebase
# @app.route('/', methods=['GET', 'POST'])
# def basic():
#     if request.method == 'POST':
#         upload = request.files['upload']
#         storage.child("images/new.jpg").put(upload)
#         return redirect(url_for('uploads'))
#     return render_template('index.html')

# Guide for get images url from firebase
# @app.route('/uploads', methods=['GET', 'POST'])
# def uploads():
#     if request.method == 'POST':
#         return redirect(url_for('basic'))
#     if True:
#         links = storage.child('images/new.jpg').get_url(None)
#         print(links)
#         return render_template('upload.html', l=links)
#     return render_template('upload.html')

@app.errorhandler(HTTPException) # if found error will response error message
def handle_exception(e):
    # start with the correct headers and status code from the error
    response = e.get_response()
    # replace the body with JSON
    response.data = json.dumps({
        "code": e.code,
        "name": e.name,
        "description": e.description,
    })
    response.content_type = "application/json"
    return response


@app.route('/', methods=['GET']) # ping
def ping():
    data = {
        "stsatus": "ok",
    }
    return Response(status=200, mimetype="application/json", response=json.dumps(data))


@app.route('/predict', methods=['POST'])
def predict():
    upload = request.files['upload'] # request key name = 'upload'
    upload.save("img.jpg")

    # Adjust the picture to be ready to predict.
    loadModel = load_model("model.h5")
    image = cv2.imread("img.jpg")
    image = cv2.resize(image, (128, 128))
    image = image.astype("float") / 255.0
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)

    # predict
    res = loadModel.predict(image)
    label = np.argmax(res)
    labelName = names[label]

    os.remove("img.jpg")

    return Response(status=200, mimetype="application/json", response=json.dumps({"name": labelName}))


if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0",port=8000)
