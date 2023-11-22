from flask import Flask, render_template, request
import numpy as np
import pickle
filename = 'diabetes-prediction-rfc-model.pkl'
classifier = pickle.load(open(filename, 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        preg = int(request.form['pregnancies'])
        glucose = int(request.form['glucose'])
        bp = int(request.form['bloodpressure'])
        st = int(request.form['skinthickness'])
        insulin = int(request.form['insulin'])
        bmi = float(request.form['bmi'])
        dpf = float(request.form['dpf'])
        age = int(request.form['age'])
        
        data = np.array([[preg, glucose, bp, st, insulin, bmi, dpf, age]])
        my_prediction = classifier.predict(data)
        if my_prediction == 1:
            pred = "You have Diabetes, please consult a Doctor."
        elif my_prediction == 0:
            pred = "You don't have Diabetes."
        output = pred

        return render_template('index.html', prediction_text='{}'.format(output))
        # return render_template('index.html', prediction=my_prediction)

if __name__ == '__main__':
	app.run(debug=True,port=3000)

#cd C:\Users\Owner\spam-env
#Set-ExecutionPolicy Unrestricted -Scope Process
#Scripts/activate
#cd C:\D_drive\coder\ml\diabetes_prediction
#python app.py