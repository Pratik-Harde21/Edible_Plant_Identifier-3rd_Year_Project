from flask import Flask, render_template, request
# from keras.applications import ResNet50
import cv2
import numpy as np
import pandas as pd
from tensorflow import keras
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import base64
import io
from PIL import Image

app = Flask(__name__)

global model
# model = load_model("model.kerasmodel")
model = load_model("mymodel.h5")

print("+"*50, "Model is loaded")

label = pd.read_fwf("label.txt").values
labels = pd.read_fwf("description.txt").values

@app.route('/')
def index():
	return render_template("index1.html", data="hey")
	# return render_template("index.html", data="hey")


@app.route("/prediction", methods=["POST"])
def prediction():

	img = request.files['img']

	img.save("img.jpg")

	image = cv2.imread("img.jpg")

	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

	image = cv2.resize(image, (224,224))

	image = np.reshape(image, (1,224,224,3))

	pred = model.predict(image)

	pred = np.argmax(pred)
	pre = model.predict(image)

	pre = np.argmax(pre)

	pred = label[pred-1]
	pre = labels[pre-1]

	return render_template("prediction.html", data=pred ,d=pre)
	# return render_template("index1.html", data=pred)


if __name__ == "__main__":
	app.run(debug=True)
