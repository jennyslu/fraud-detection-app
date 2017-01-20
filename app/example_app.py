from flask import Flask, request, render_template, jsonify
import requests
import socket
import time
from datetime import datetime
from feature_format import feature_engineering
import json
import pickle
from pymongo import MongoClient

client = MongoClient()
db = client['fraud_case_study']
coll = db['new']

app = Flask(__name__)
DATA = []
TIMESTAMP = []
CLF_FILE_NAME = "classifier.pkl"
LINK = 'http://galvanize-case-study-on-fraud.herokuapp.com/data_point'

with open(CLF_FILE_NAME, "rb") as clf_infile:
    MODEL = pickle.load(clf_infile)

def scrape():
    response = requests.get(LINK)
    return response.json()

# My word counter app
@app.route('/')
def make_prediction():
    new_data = scrape()
    DATA.append(new_data)
    X, feature_names = feature_engineering(DATA, [])
    proba = MODEL.predict_proba(X[-1].reshape(1,X.shape[1]))[:,1][0]
    probability = "{:.2%}".format(proba)
    if proba < 0.2:
        fraud_text = "Not Fraud!"
    else:
        fraud_text = "Fraud!"
    return render_template('index.html', prediction=fraud_text,
                            proba=probability, event_data=new_data)
def make_predictions():
    new_data = scrape()
    DATA.append(new_data)
    X, feature_names = feature_engineering(DATA, [])
    proba = MODEL.predict_proba(X[-1].reshape(1,X.shape[1]))[:,1][0]
    probability = "{:.2%}".format(proba)
    if proba < 0.2:
        fraud_text = "Not Fraud!"
    else:
        fraud_text = "Fraud!"
    document = {"_id":new_data['object_id'],
                "response":new_data,
                "prediction":fraud_text,
                "probability":proba}
    try:
        coll.insert_one(document)
    except:
        pass
    return fraud_text, probability, new_data


@app.route('/_refresh')
def refresh():
    fraud_text, probability, new_data = make_predictions()
    result = {"fraud_text": fraud_text,
                "probability": probability,
                "new_data": new_data}
    return jsonify(result)


@app.route('/check')
def check():
    line1 = "Number of data points: {0}".format(len(DATA))
    if DATA and TIMESTAMP:
        dt = datetime.fromtimestamp(TIMESTAMP[-1])
        data_time = dt.strftime('%Y-%m-%d %H:%M:%S')
        line2 = "Latest datapoint received at: {0}".format(data_time)
        line3 = DATA[-1]
        output = "{0}\n\n{1}\n\n{2}".format(line1, line2, line3)
    else:
        output = line1
    return output, 200, {'Content-Type': 'text/css; charset=utf-8'}



if __name__ == '__main__':
    # Start Flask app
    app.run(host='0.0.0.0', port=8080, debug=True)
