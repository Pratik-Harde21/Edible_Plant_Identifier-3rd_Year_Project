from flask import Flask, request, render_template, jsonify
from tensorflow import keras
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import base64
import numpy as np
import io
from PIL import Image

app = Flask(__name__)

categories=['alfalfa',
'allium',
'borage',
'burdock',
'calendula',
'cattail',
'chickweed',
'chicory',
'chive_blossom',
'coltsfoot',
'common_mallow',
'common_milkweed',
'common_vetch',
'common_yarrow',
'coneflower',
'cow_parsley',
'cowslip',
'crimson_clover',
'crithmum_maritimum',
'daisy',
'dandelion',
'fennel',
'fireweed',
'gardenia',
'garlic_mustard',
'geranium',
'ground_ivy',
'harebell',
'henbit',
'knapweed',
'meadowsweet',
'mullein',
'pickerelweed',
'ramsons',
'red_clover']

def get_model():
    global model
    # model = load_model("model.kerasmodel")
    model = load_model("mymodel.h5")
    print(" * Model loaded!")


def preprocess_image(image, target_size):
    if image.mode != "RGB":
        image = image.convert("RGB")
        
    image = image.resize(target_size)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    return image


print(" * Loading Keras model...")
get_model()


@app.route("/")
def index():
    return render_template('index.html')


@app.route('/home')
def home():
    return render_template('predict.html')


@app.route("/predict", methods=["POST"])
def predict():
    # return render_template('predict.html')
    message = request.get_json(force=True)
    encoded = message['image']
    decoded = base64.b64decode(encoded)
    image = Image.open(io.BytesIO(decoded))
    processed_image = preprocess_image(image, target_size=(224, 224))
    # prediction = model.predict(processed_image).tolist()
    prediction = model.predict(processed_image)

    # response = {
    #     'prediction': {
    #         'pred': prediction,
    #         'alfalfa': prediction[0][0],
    #         'allium': prediction[0][1],
    #         'borage': prediction[0][2],
    #         'burdock': prediction[0][3],
    #         'calendula': prediction[0][4],
    #         'cattail': prediction[0][5],
    #         'chickweed': prediction[0][6],
    #         'chicory': prediction[0][7],
    #         'chive_blossom': prediction[0][8],
    #         'coltsfoot': prediction[0][9],
    #         'common_mallow': prediction[0][10],
    #         'common_milkweed': prediction[0][11],
    #         'common_vetch': prediction[0][12],
    #         'common_yarrow': prediction[0][13],
    #         'coneflower': prediction[0][14],
    #         'cow_parsley': prediction[0][15],
    #         'cowslip': prediction[0][16],
    #         'crimson_clover': prediction[0][17],
    #         'crithmum_maritimum': prediction[0][18],
    #         'daisy': prediction[0][19],
    #         'dandelion': prediction[0][20],
    #         'fennel': prediction[0][21],
    #         'fireweed': prediction[0][22],
    #         'gardenia': prediction[0][23],
    #         'garlic_mustard': prediction[0][24],
    #         'geranium': prediction[0][25],
    #         'ground_ivy': prediction[0][26],
    #         'harebell': prediction[0][27],
    #         'henbit': prediction[0][28],
    #         'knapweed': prediction[0][29],
    #         'meadowsweet': prediction[0][30],
    #         'mullein': prediction[0][31],
    #         'pickerelweed': prediction[0][32],
    #         'ramsons': prediction[0][33],
    #         'red_clover': prediction[0][34]
    #     }
    # }
    return render_template("predict.html", data=prediction)
    # return jsonify(response)


if __name__ == "__main__":
    app.run()
