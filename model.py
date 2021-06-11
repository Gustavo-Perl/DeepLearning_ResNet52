import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import decode_predictions, ResNet50

model = ResNet50(weights='imagenet')
def predict(image_file):
    prediction  = model.predict(image_file)
    classe      = decode_predictions(prediction, top = 1)[0][0][1]
    proba       = round(decode_predictions(prediction, top = 1)[0][0][2] * 100,2)
    return [classe, proba]