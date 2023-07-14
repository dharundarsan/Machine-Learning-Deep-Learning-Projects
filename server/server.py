from ctypes import util
from flask import Flask, request, jsonify
import utils
app = Flask(__name__)

@app.route('/get_loc_name')
def get_loc_name():
    response = jsonify({
        'locations' : utils.get_location_names()
    })

    response.headers.add('Access-Control-Allow-Origin', '*')

    return response

@app.route('/predict_price', methods = ['POST'])
def predict_price():
    total_sqft = float(request.form['total_sqft'])
    location = (request.form['location'])
    bhk = float(request.form['bhk'])
    bath = float(request.form['bath'])

    response = jsonify({
        'estimated_price' : utils.get_estimated_price(location, total_sqft, bhk, bath)

    })

    response.headers.add("Access-Control-Allow-Origin", "*")

    return response


if __name__ == "__main__":
    print("praticing")
    utils.load_saved_artifacts()
    app.run() 