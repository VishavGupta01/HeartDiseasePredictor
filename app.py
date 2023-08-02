from flask import Flask,request,jsonify
import pickle
import numpy as np

model = pickle.load(open('model.pkl','rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return "hello World"

@app.route('/predict', methods=['POST'])
def predict():
    Age = int(request.form.get('Age'))
    Sex = int(request.form.get('Sex'))
    ChestPainType = int(request.form.get('ChestPainType'))
    RestingBP = int(request.form.get('RestingBP'))
    Cholesterol = int(request.form.get('Cholesterol'))
    FastingBS = int(request.form.get('FastingBS'))
    RestingECG = int(request.form.get('RestingECG'))
    MaxHR = int(request.form.get('MaxHR'))
    ExerciseAngina = int(request.form.get('ExerciseAngina'))
    Oldpeak = float(request.form.get('Oldpeak'))
    ST_Slope = int(request.form.get('ST_Slope'))
    #HeartDisease = int(request.form.get('HeartDisease'))

   # result = { 'Age' :Age , 'Sex' :Sex ,'ChestPainType' :ChestPainType ,'RestingBP' :RestingBP ,'Cholesterol' :Cholesterol ,'FastingBS' :FastingBS ,'RestingECG' :RestingECG ,'MaxHR' :MaxHR ,'ExerciseAngina' :ExerciseAngina , 'Oldpeak' :Oldpeak , 'ST_Slope' :ST_Slope , 'HeartDisease' :HeartDisease }
    input_query = np.array([[Age,Sex,ChestPainType,RestingBP,Cholesterol,FastingBS,RestingECG,MaxHR,ExerciseAngina,Oldpeak,ST_Slope]])

    result = model.predict(input_query)[0]

    return jsonify({'disease': str(result)})

if __name__ == '__main__':
    app.run(debug=True)