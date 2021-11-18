import os
import pandas as pd
from flask import Flask, request, make_response, jsonify

from predictor import PredictorService
from pyschemavalidator import validate_param
import logging

# Logger creation
logging.basicConfig(level=logging.DEBUG, format="[%(asctime)s] %(levelname)s - %(message)s")
logger = logging.getLogger()

# The flaks app to serve predictions
app = Flask(__name__)

PREFIX = "/opt/ml/"
MODEL_PATH = os.path.join(PREFIX, "model")
model = PredictorService(MODEL_PATH)
model.start()

THRESHOLD = 0.5


def health():
    """
    Sanity check to make sure the container is properly running.
    """
    return make_response("", 200)

@app.route("/health", methods=["GET"])
def home():
    return health()

@app.route("/ping", methods=["GET"])
def ping():
    return health()

@app.route("/invocations", methods=["POST"])
@validate_param(key="taxi_ride_id", keytype=int, isrequired=True)
@validate_param(key="pickup_datetime", keytype=str, isrequired=True)
@validate_param(key="pickup_longitude", keytype=float, isrequired=True)
@validate_param(key="pickup_latitude", keytype=float, isrequired=True)
@validate_param(key="dropoff_longitude", keytype=float, isrequired=True)
@validate_param(key="dropoff_latitude", keytype=float, isrequired=True)
@validate_param(key="passenger_count", keytype=int, isrequired=True)
def invocations():
    """
        Online prediction on single data instance. Data is accepted as JSON and then properly parsed.
        Then, the model predicts taxi fare amount.
    """
    data = request.get_json(silent=True)
    df = pd.DataFrame([data])
    pred = model.predict(df)[0]
    pred = round(pred, 2)
    return make_response(
        jsonify({
            "taxi_ride_id": data["taxi_ride_id"],
            "taxi_fare_amount": pred
        }),
        200
    )