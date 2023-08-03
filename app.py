from flask import Flask, request, render_template, url_for, jsonify
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

app = Flask(__name__)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# def preprossing(image):
#     if image.mode != 'RGB':
#         image = image.convert('RGB')

#     image = Image.open(image)
#     image = image.resize((224, 224))  # Ubah ukuran menjadi 224x224
#     image_arr = np.array(image.convert('RGB'))
#     image_arr.shape = (1, 224, 224, 3)
#     return image_arr

from io import BytesIO

def preprocessing(image):
    image = Image.open(image)
    if image.mode != 'RGB':
        image = image.convert('RGB')

    image = image.resize((224, 224))  # Ubah ukuran menjadi 224x224
    image_arr = np.array(image)
    image_arr.shape = (1, 224, 224, 3)
    return image_arr


classes = ['NORMAL', 'PNEUMONIA']
model=load_model("chest_xray.h5")

@app.route('/')
def index():

    return render_template('index.html', appName="Intel Image Classification")

# dirubah
@app.route('/predictApi', methods=["POST"])
def api():
    # Get the image from post request
    try:
        # image = request.files.get('fileup')
        if 'fileup' not in request.files:
            return "Please try again. The Image doesn't exist"
        image = request.files.get('fileup')
        image_arr = preprocessing(image)
        print("Model predicting ...")
        result = model.predict(image_arr)
        print("Model predicted")
        ind = np.argmax(result)
        prediction = classes[ind]
        print(prediction)
        return jsonify({'prediction': prediction})
    except:
        return jsonify({'Error': 'Error occur'})


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    print("run code")
    if request.method == 'POST':
        # Get the image from post request
        print("image loading....")
        image = request.files['fileup']
        print("image loaded....")
        image_arr= preprocessing(image)
        print("predicting ...")
        result = model.predict(image_arr)
        print("predicted ...")
        ind = np.argmax(result)
        prediction = classes[ind]

        print(prediction)

        return render_template('index.html', prediction=prediction, image='js/IMG/', appName="Intel Image Classification")
    else:
        return render_template('index.html',appName="Intel Image Classification")


if __name__ == '__main__':
    app.run(debug=True)