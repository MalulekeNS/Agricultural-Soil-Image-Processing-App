from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__)

dependencies = {
    'auc_roc': tf.keras.metrics.AUC
}

verbose_name = {
    0: 'Alluvial_Soil (This soil is suitable for the crops:- { Rice, Wheat, Sugarcane, Maize, Cotton, Soyabean, Jute })',
    1: 'Black_Soil (This soil is suitable for the crops:- { Virginia, Wheat, Jowar, Millets, Linseed, Castor, Sunflower })',
    2: 'Clay_Soil (This soil is suitable for the crops:- { Rice, Lettuce, Chard, Broccoli, Cabbage, Snap Beans })',
    3: 'Red_Soil (This soil is suitable for the crops:- { Cotton, Wheat, Pulses, Millets, OilSeeds, Potatoes })'
}

# Select model
model = load_model('soil_multi_output_model.h5', compile=False)

model.make_predict_function()

def predict_label_and_ph(img_path):
    test_image = image.load_img(img_path, target_size=(128, 128))
    test_image = image.img_to_array(test_image) / 255.0
    test_image = test_image.reshape(1, 128, 128, 3)

    predictions = model.predict(test_image)
    soil_type_index = np.argmax(predictions[0], axis=1)[0]  # Soil type prediction
    ph_value = predictions[1][0][0]  # pH value prediction

    soil_type = verbose_name[soil_type_index]
    ph_value = round(ph_value, 2)

    return soil_type, ph_value

# Routes

@app.route("/")
@app.route("/first")
def first():
    return render_template('first.html')

@app.route("/login")
def login():
    return render_template('login.html')

@app.route("/index", methods=['GET', 'POST'])
def index():
    return render_template("index.html")

@app.route("/submit", methods=['GET', 'POST'])
def get_output():
    if request.method == 'POST':
        img = request.files['my_image']
        img_path = "static/tests/" + img.filename
        img.save(img_path)

        soil_type, ph_value = predict_label_and_ph(img_path)
        prediction = {
            "soil_type": soil_type,
            "ph_value": ph_value
        }

    return render_template("prediction.html", prediction=prediction, img_path=img_path)

@app.route("/performance")
def performance():
    return render_template('performance.html')

@app.route("/chart")
def chart():
    return render_template('chart.html')

if __name__ == '__main__':
    app.run(debug=True)
