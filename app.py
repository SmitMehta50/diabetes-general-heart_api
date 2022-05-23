from flask import Flask, render_template, request, jsonify
import numpy as np
from sklearn.metrics import confusion_matrix
import joblib
import argparse
from flask_cors import CORS
import pickle


app = Flask(__name__)
CORS(app)

# **********************General Prediction Functions*************************

s = ['Consult nearest hospital',
     'acetaminophen',
     'anti boitic therapy',
     'anti itch medicine',
     'apply calamine',
     'avoid abrupt head movment',
     'avoid cold food',
     'avoid fatty spicy food',
     'avoid lying down after eating',
     'avoid nonveg food',
     'avoid oily food',
     'avoid opencuts',
     'avoid public places',
     'avoid sudden change in body',
     'avoid too many products',
     'bath twice',
     'call ambulance',
     'check impulse',
     'chew or swallow asprin',
     'cold baths',
     'consult doctor',
     'consult nearest hospital',
     'consume alovera juice',
     'consume milk thistle',
     'consume neem leaves',
     'consume probiotic food',
     'consume witch hazel',
     'cover area with bandage',
     'cover mouth',
     'dont stand still for long',
     'drink cranberry juice',
     'drink papaya leaf juice',
     'drink plenty of water',
     'drink sugary drinks',
     'drink vitamin crich drinks',
     'easeback into eating',
     'eat fruits and high fiberous food',
     'eat healthy',
     'eat high calorie vegetables',
     'eliminate milk',
     'exercise',
     'follow up',
     'get away from trigger',
     'get propersleep',
     'have balanced diet',
     'increase vitamin c intake',
     'keep calm',
     'keep fever incheck',
     'keep hydrated',
     'keep infected area dry',
     'keep mosquitos away',
     'keep mosquitos out',
     'lie down',
     'lie down flat and raise the leg high',
     'lie down on side',
     'limit alcohol',
     'maintain healthy weight',
     'massage',
     'medication',
     'meditation',
     'reduce stress',
     'relax',
     'remove scabs with wet compressed cloth',
     'rest',
     'salt baths',
     'seek help',
     'soak affected area in warm water',
     'stop alcohol consumption',
     'stop bleeding using pressure',
     'stop eating solid food for while',
     'stop irritation',
     'stop taking drug',
     'switch to loose cloothing',
     'take deep breaths',
     'take otc pain reliver',
     'take probiotics',
     'take radioactive iodine treatment',
     'take vaccine',
     'take vapour',
     'try acupuncture',
     'try taking small sips of water',
     'use antibiotics',
     'usecle ancloths',
     'use detol or neem in bathing water',
     'use heating pad or coldpack',
     'use hot and coldtherapy',
     'use ice to compress itching',
     'use lemon balm',
     'use neem in bathing',
     'use oinments',
     'use poloroid glasses in sun',
     'use vein compression',
     'vaccination',
     'warm bath with epsom salt',
     'wash hands through',
     'wash hands with warm soapy water',
     'wear ppe if possible']


with open('Symptom.pkl', 'rb') as f:
    d1 = pickle.load(f)

with open('Symptom_Description.pkl', 'rb') as f:
    d2 = pickle.load(f)

with open('Symptom_precaution.pkl', 'rb') as f:
    d3 = pickle.load(f)

t = [j for i in list(d3.values()) for j in i]
t = np.unique(t).tolist()
t = dict(zip(t, s))

with open('Symptoms_list.pkl', 'rb') as f:
    l = pickle.load(f)

model = joblib.load('model.joblib')
mlb = joblib.load('mlb.joblib')


def output(l, d1):
    d = {}
    for i in l:
        if i in d1:
            d[i] = d1[i]

    return d


def predict(sym):
    data = mlb.transform([sym])
    pred = model.predict(data)[0]
    s1 = output(sym, d1)
    # s1 = {k: v for k, v in sorted(
    #     s1.items(), key=lambda item: item[1], reverse=True)}
    s2 = output([pred], d2)
    s3 = output([pred], d3)
    s3[list(s3.keys())[0]] = [t[i][0].upper()+t[i][1:]
                              for i in s3[list(s3.keys())[0]]]
    return pred, s1, s2, s3


@app.route('/', methods=['GET'])
def startpage():
    return jsonify({
        "URLs": [{"diabetes": "https://diabetessapi.herokuapp.com/diabetes"},
                 {"general prediction": "https://diabetessapi.herokuapp.com/generalprediction"},
                 {"heart": "https://diabetessapi.herokuapp.com/heart"},
                 {"pneumonia": "https://pneumoniaapi.herokuapp.com/"},
                 {"kidney": "https://kidneydisease-api.herokuapp.com/kidney"},
                 {"liver": "https://kidneydisease-api.herokuapp.com/liver"}]
    })


# ***************************Diabetes Route**********************************

@app.route('/diabetes', methods=['GET', 'POST'])
def diabetes():
    if request.method == 'GET':
        return jsonify({'message': "This gives diabetes prediction"},
                       {"POST-Key for uploading data": [
                           {"Pregnancies": "Pregnancies"},
                           {"Glucose": "Glucose"},
                           {"BloodPressure": "BloodPressure"},
                           {"SkinThickness": "SkinThickness"},
                           {"Insulin": "Insulin"},
                           {"BMI": "BMI"},
                           {"Diabetes Pedigree Function": "DiabetesPedigreeFunction"},
                           {"Age": "Age"}
                       ]})

    elif request.method == 'POST':
        scaler = joblib.load('scaler.joblib')
        model = joblib.load('Diabetes_model.joblib')

        request_data = request.get_json()
        Pregnancies = request_data['Pregnancies']
        Glucose = request_data['Glucose']
        BloodPressure = request_data['BloodPressure']
        SkinThickness = request_data['SkinThickness']
        Insulin = request_data['Insulin']
        Bmi = request_data['BMI']
        DiabetesPedigreeFunction = request_data['DiabetesPedigreeFunction']
        Age = request_data['Age']

        data = np.array([[Pregnancies, Glucose, BloodPressure,
                        SkinThickness, Insulin, Bmi, DiabetesPedigreeFunction, Age]])
        data = scaler.transform(data)
        diabetes_prediction = int(model.predict(data)[0])
        prob_diabetes = round((model.predict_proba(data)[0][1])*100)
        print(prob_diabetes)
        print(diabetes_prediction)

        if diabetes_prediction == 1:
            return jsonify({'message': 'Diabetes Detected', 'probability': prob_diabetes})
        elif diabetes_prediction == 0:
            return jsonify({'message': 'Diabetes Not Detected', 'probability': prob_diabetes})

        # return jsonify({'message': diabetes_prediction, 'probability': prob_diabetes})
        # return render_template("diabetes.html", diabetes_prediction=diabetes_prediction, prob_diabetes=prob_diabetes)

# **********************General Prediction Route********************************


@app.route('/generalprediction', methods=['GET', 'POST'])
def general_predict():
    if request.method == 'GET':
        return jsonify({'message': "Genral disease prediction"},
                       {"POST-Key for uploading data": "Return it as list data"})

    elif request.method == 'POST':

        request_data = request.get_json()
        # sympton1 = request_data['sympton1']
        # sympton2 = request_data['sympton2']
        # sympton3 = request_data['sympton3']
        # sympton4 = request_data['sympton4']
        # sympton5 = request_data['sympton5']

        # sym = [sympton1, sympton2, sympton3, sympton4, sympton5]
        sym = request_data
        print(sym)
        pred, s1, s2, s3 = predict(sym)

        # print(pred)
        # print(s1)
        # print(s2)
        # print(s3)
        # print(l)
        return jsonify({'prediction': pred}, {'rate': s1}, {'discription': s2[pred]}, {'treatment': s3[pred]})


# **********************Heart Disease Route******************
@app.route('/heart', methods=['GET', 'POST'])
def heart():
    if request.method == 'GET':
        return jsonify({'message': "This gives heart disese prediction"},
                       {"POST-Key for uploading data": [
                           {"age": "age",
                            "sex": "sex",
                            "cp": "cp",
                            "trestbps": "trestbps",
                            "chol": "chol",
                            "fbs": "fbs",
                            "restecg": "restecg",
                            "thalach": "thalach",
                            "exang": "exang",
                            "oldpeak": "oldpeak",
                            "slope": "slope",
                            "ca": "ca",
                            "thal": "thal"}
                       ]})

    elif request.method == 'POST':
        model = joblib.load('model.pkl')
        scaler = joblib.load('scaler.pkl')

        def predict(data):
            if data[1] == 'Female':
                data[1] = 0
            else:
                data[1] = 1
            data = scaler.transform([data])
            pred = model.predict_proba(data)
            return pred.argmax(), pred.ravel()[pred.argmax()]

        request_data = request.get_json()
        age = request_data['age']
        sex = request_data['sex']
        cp = request_data['cp']
        trestbps = request_data['trestbps']
        chol = request_data['chol']
        fbs = request_data['fbs']
        restecg = request_data['restecg']
        thalach = request_data['thalach']
        exang = request_data['exang']
        oldpeak = request_data['oldpeak']
        slope = request_data['slope']
        ca = request_data['ca']
        thal = request_data['thal']

     # data = [eval(i) for i in col]
        data = [age, sex, cp, trestbps, chol, fbs, restecg,
                thalach, exang, oldpeak, slope, ca, thal]
        pred, prob = predict(data)

        if pred == 1:
            return jsonify({'message': 'Heart Disease Detected', 'probability': prob})
        elif pred == 0:
            return jsonify({'message': 'Heart Diaease Not Detected', 'probability': prob})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--port", type=int, default=8080)
    args = parser.parse_args()
    app.run(debug=True, port=args.port)
