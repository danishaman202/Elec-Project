# -*- coding: utf-8 -*-
"""
Created on Wed Feb  2 19:16:04 2022

@author: user
"""
import numpy as np
from flask import Flask, request, jsonify, render_template
import joblib

app = Flask(__name__) #Initialize the flask App
model = joblib.load('Isolationforest.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [x for x in request.form.values()]
    tester = np.array([float(int_features[3]),float(int_features[4]),float(int_features[5])]).reshape(1,-1)
    
    prediction = model.predict(tester)
    
    if prediction == -1:
        if ((int(int_features[3])) > 220 and (int(int_features[4])) > 220 and (int(int_features[5])) > 220):
            ret = "input is AbnormalThe issue causing this abnormality are surges and are at Location",int_features[7]
            return render_template('index.html', prediction_text=ret)
        else:
            ret = "input is AbnormalThe issue causing this abnormality are sags and are at Location",int_features[7]
            return render_template('index.html', prediction_text=ret)
    else:
        return render_template('index.html', prediction_text="Normal Values")
    

if __name__ == "__main__":
    app.run(debug=True)
