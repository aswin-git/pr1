from flask import Flask, request, jsonify, Response
import logging
from prometheus_client import  generate_latest, CONTENT_TYPE_LATEST
from monitoring import REQUEST_COUNT, REQUEST_LATENCY
from logging_config import setup_logging
import mlflow
import numpy as np

app = Flask(__name__)

setup_logging()

model = mlflow.pyfunc.load_model('mlflow/0/models/m-07f50c975067406d8c48f13132d1220d/artifacts')

@app.route('/predict', methods = ['POST'])
def predict():
    logging.info('Request recived')

    REQUEST_COUNT.inc()
    data = request.get_json()
    value = np.array(data).reshape(1,-1)

    with REQUEST_LATENCY.time():
        prediction = model.predict(value)

    logging.info(f'input: {value} , predivtion - {prediction}')

    return jsonify({'prediction': int(prediction[0])})
@app.route('/metrics', methods = ['GET'])
def metrics():
    return Response(generate_latest(),  mimetype=CONTENT_TYPE_LATEST)

 
app.run(host='0.0.0.0',port=5050)