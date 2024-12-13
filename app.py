from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)
model = pickle.load(open('flight_model.pkl', 'rb'))  

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    airline = float(request.form['airline'])
    flight = float(request.form['flight'])
    source_city = float(request.form['source_city'])
    departure_time = float(request.form['departure_time'])
    stops = float(request.form['stops'])
    arrival_time = float(request.form['arrival_time'])
    destination_city = float(request.form['destination_city'])
    flight_class = float(request.form['flight_class'])
    duration = float(request.form['duration'])
    days_left = float(request.form['days_left'])

    user_inputs=np.array([[airline,flight,source_city,departure_time,stops,arrival_time,destination_city,flight_class,duration,days_left]])
    p=model.predict(user_inputs)
    return render_template('result.html', prediction=p[0])


if __name__ == '__main__':
    app.run(debug=True)
