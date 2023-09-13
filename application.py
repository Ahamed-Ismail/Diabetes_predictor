from flask import Flask, request, app,render_template
from flask import Response
import pickle
import numpy as np
import pandas as pd


application = Flask(__name__)
app=application


#importing model
scaler=pickle.load(open("models/scaler.pkl", "rb"))
model = pickle.load(open("models/DecisionTree.pkl", "rb"))   #has the best accuracy

@app.route('/', methods=['POST','GET'])
def home():

    if request.method=='POST':

        Pregnancies=int(request.form.get("Pregnancies"))
        Glucose = float(request.form.get('Glucose'))
        BloodPressure = float(request.form.get('BloodPressure'))
        SkinThickness = float(request.form.get('SkinThickness'))
        Insulin = float(request.form.get('Insulin'))
        BMI = float(request.form.get('BMI'))
        DiabetesPedigreeFunction = float(request.form.get('DiabetesPedigreeFunction'))
        Age = float(request.form.get('Age'))

        new_data=scaler.transform([[Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]])
        predict=model.predict(new_data)

        if predict[0]==0:
            result="Non Diabetic"
        else:
            result='Diabetic'
        
        return render_template('index.html', result=result)
    else:
        return render_template('index.html')
    

if __name__=='__main__':
    app.run('0.0.0.0',port=3000)