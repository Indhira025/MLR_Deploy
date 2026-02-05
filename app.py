from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load trained ML model
model = pickle.load(open('MLR_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input values from form
        rd = float(request.form['RD'])
        admin = float(request.form['Admin'])
        marketing = float(request.form['Marketing'])
        state = int(request.form['State'])

        # Prepare input for model
        features = np.array([[rd, admin, marketing, state]])

        # Make prediction
        prediction = model.predict(features)

        # Format output
        output = round(prediction[0], 2)

        return render_template(
            'index.html',
            prediction_text=f"₹ {output}"
        )

    except Exception as e:
        return render_template(
            'index.html',
            prediction_text="Invalid input, please try again"
        )

if __name__ == "__main__":
    app.run(debug=True)
