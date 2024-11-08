import joblib
import numpy as np
from flask import Flask, request, render_template

#Initialize flask app
app = Flask(__name__)

#Load the trained model
model = joblib.load('model.pkl')


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            #Get form data
            gender = int(request.form['gender'])
            married = int(request.form['married'])
            dependents = int(request.form['dependents'])
            education = int(request.form['education'])
            self_employed = int(request.form['self_employed'])
            applicant_income = float(request.form['applicant_income'])
            coapplicant_income = float(request.form['coapplicant_income'])
            loan_amount = float(request.form['loan_amount'])
            loan_amount_term = float(request.form['loan_amount_term'])
            credit_history = float(request.form['credit_history'])
            property_area = int(request.form['property_area'])

            #Array of input feature 
            features = np.array([[gender, married, dependents, education, self_employed,
                                  applicant_income, coapplicant_income, loan_amount,
                                  loan_amount_term, credit_history, property_area,]])

            #Make predictions
            prediction = model.predict(features)[0]
            result = 'Approved' if prediction == 1 else 'Rejected'

            return render_template('index.html', result=result)

        except Exception as e:
            return render_template('index.html', result="Error: {}".format(e))
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
