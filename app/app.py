import matplotlib
matplotlib.use('Agg')   # ⭐ IMPORTANT (fix error)

import matplotlib.pyplot as plt
import os
from flask import Flask, render_template, request
import pickle
import numpy as np

# App create
app = Flask(__name__)

# Model load
model = pickle.load(open('../model/model.pkl', 'rb'))

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 🔹 Input features
        features = [
            float(request.form['semester']),
            float(request.form['cgpa']),
            float(request.form['attendance']),
            float(request.form['aptitude']),
            float(request.form['coding']),
            float(request.form['communication']),
            float(request.form['projects']),
            float(request.form['internships'])
        ]

        final_features = [np.array(features)]

        # 🔹 Prediction
        prediction = model.predict(final_features)

        # 🔹 Placement probability
        prob = model.predict_proba(final_features)[0][1]
        chance = round(prob * 100, 2)

        # 🎯 GRAPH CODE
        labels = ['CGPA', 'Aptitude', 'Coding', 'Communication']
        user_scores = [
            features[1],
            features[3],
            features[4],
            features[5]
        ]

        ideal_scores = [8, 80, 80, 75]

        plt.figure(figsize=(6,4))
        plt.plot(labels, user_scores, marker='o', label='Your Score')
        plt.plot(labels, ideal_scores, marker='o', label='Ideal Score')
        plt.legend()
        plt.title("Performance Graph")
        plt.grid(True)

        os.makedirs('static/images', exist_ok=True)
        plt.savefig('static/images/graph.png')
        plt.close()

        # 🎯 SUGGESTION CODE
        suggestion = ""

        # CGPA
        if features[1] < 7:
            suggestion += "Improve your CGPA. "

        # Coding
        if features[4] < 70:
            suggestion += "Improve coding skills. "

        #Communication
        if features[5] < 60:
            suggestion += "Work on communication skills. "

        # Aptitude
        if features[3] < 60:
            suggestion += "Practice aptitude regularly. "

        # Projects (NEW)
        if features[6] < 2:
            suggestion += "Increase projects. "

        # If everything is good
        if suggestion == "":
            suggestion = "Great! You are on the right track 🚀"

        # 🔹 Final Output
        output = "Placed ✅" if prediction[0] == 1 else "Not Placed ❌"

        return render_template(
            'index.html',
            prediction_text=output,
            suggestion=suggestion,
            chance=chance
        )

    except Exception as e:
        return str(e)

# Run app
if __name__ == "__main__":
    app.run(debug=True)