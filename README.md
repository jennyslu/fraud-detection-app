# fraud-detection-app

## Run:

1. `cd app/` and `python example_app.py`
2. Navigate to localhost:8080 (or 0.0.0.0:8080) in your browser

## Requirements:

- Python
- Flask
- MongoDB (optional, can comment out data storage)

## Description:

This web app simulates a real-time fraud detection app for a ticket sales company (e.g. StubHub). 

Classifier is a RandomForestClassifier. For more details on feature engineering please inspect files in main folder. 

Final test scores: 

 - Precision: 92.9%
 
 - Recall: 88.1%
 
 - F1: 90.5%
