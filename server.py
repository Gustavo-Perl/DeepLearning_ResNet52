from flask import Flask, request, jsonify
import cv2
import numpy as np
import model
import tensorflow
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions

app = Flask(__name__)
@app.route('/api', methods = ['POST'])

def predict():    
    nparr            = np.fromstring(request.data, np.uint8) # convert string of image data to uint8
    img              = cv2.imdecode(nparr, cv2.IMREAD_COLOR)# decode image
    img              = cv2.resize(img, (224, 224), interpolation = cv2.INTER_AREA)
    img_batch        = np.expand_dims(img, axis = 0)
    img_preprocessed = preprocess_input(img_batch)
    prediction       = model.predict(img_preprocessed)
    return jsonify(prediction)

if __name__ == '__main__':
    app.run(debug = False, threaded=False)

