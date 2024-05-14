from flask import Flask, render_template, request, redirect, url_for, session, flash
from functools import wraps
import pandas as pd
import numpy as np
import os
import pickle
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime, timedelta
import calendar

app = Flask(__name__)
app.secret_key = 'ZiyiChen'
app.config['SESSION_PERMANENT'] = False
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(hours=1)

users = {
    'masta':{
        'username': 'masta',
        'email': 'masta123@gmail.com',
        'password': generate_password_hash('123')
    }
}

# This function is checking whether the system has logined
def login_check(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user' not in session:
            return redirect(url_for('home', login_required='1'))
        return f(*args, **kwargs)
    return decorated_function

# This is the home page
@app.route('/')
def home():
    user = session.get('user', None)
    login_required = request.args.get('login_required', None)
    login_error = "You need to login first." if login_required else None
    return render_template('home.html', user=user, date=datetime.now(), login_error=login_error)

# This is about the prediction
@app.route('/predict', methods=['GET', 'POST'])
@login_check
def predict():
    if request.method == 'POST':
        model = request.form['mode']
        condition = request.form['THO']
        condition_min = request.form['minValue']
        condition_max = request.form['maxValue']

        parameters = request.form.getlist('parameter[]')
        values = request.form.getlist('parameterValue[]')

        prediction_data = {
            'Model': model,
            'Condition': {
                'condition': condition,
                'condition_min': condition_min,
                'condition_max': condition_max
            },
            'Parameters': []
        }
        
        for param, val in zip(parameters, values):
            prediction_data['Parameters'].append({
                'parameter': param,
                'value': val
            })
        session['prediction_data'] = prediction_data
        session.modified = True
        
        return redirect(url_for('result'))
    return render_template('predict.html')

# This is the history page
@app.route('/history')
@login_check
def history():
    history_data = session.get('history', [])
    return render_template('history.html', history_data=history_data)

@app.route('/about')
def about():
    return render_template('about.html')

# This is about the login, log out and register:
@app.route('/login', methods=['GET','POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = users.get(username)
        if user and check_password_hash(user['password'], password):
            session['user'] = username
            return redirect(url_for('home'))
        else:
            error = 'Invalid username or password'
            return render_template('home.html', login_error=error)
    return redirect(url_for('home'))

@app.route('/logout', methods=['POST'])
def logout():
    session.clear()
    session.pop('user', None)
    return redirect(url_for('home'))

@app.route('/register', methods=['GET','POST'])
def register():
    if request.method == 'POST':
        username = request.form['new_username']
        email = request.form['email']
        password = generate_password_hash(request.form['new_password'])
        for user in users.values():
            if user['username'] == username:
                return render_template('home.html', register_error='Username already exists')
            if user['email'] == email:
                return render_template('home.html', register_error='Email already exists')
        users[username] = {'username': username, 'email': email, 'password': password}
        return redirect(url_for('home'))
    return redirect(url_for('home'))

# This shows the result
@app.route('/result')
def result():
    prediction_data = session.get('prediction_data', {})
    
    file_name = 'north_wing_2019.csv' if prediction_data['Model'] == 'North' else 'south_wing_2019.csv'
    file_path = os.path.join(app.root_path, 'static', 'dataset', file_name)
    output = 'hvac_N' if prediction_data['Model'] == 'North' else 'hvac_S'
    model_name = 'rfr_model_north.pkl' if prediction_data['Model'] == 'North' else 'rfr_model_south.pkl'
    model_path = os.path.join(app.root_path, 'static', 'model', model_name)
    try:
        df = pd.read_csv(file_path)
        condition = prediction_data['Condition']
        if condition['condition'] == "Temperature":
            con = 'temperature'
        elif condition['condition'] == "Humidity":
            con = 'humidity'
        df = df[(df[con] >= int(condition['condition_min'])) & 
                (df[con] <= int(condition['condition_max']))]

        for param in prediction_data['Parameters']:
            df[param['parameter']] = float(param['value'])

        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        features = df[model.feature_names_in_]
        hvac_pred = model.predict(features)

        df['hvac'] = hvac_pred
        df_group_month = df.groupby('month').sum().reset_index()
        hvac_pred = list(df_group_month['hvac'])
        date_list = [calendar.month_name[month] for month in df_group_month['month'] ]

        prediction_data['hvac_pred'] = f"{np.mean(hvac_pred):.2f}"
        session['prediction_data'] = prediction_data
        if 'history' not in session:
            session['history'] = []
        session['history'].append(prediction_data)
        session.modified = True

        return render_template('result.html', hvac_type=output, date_list=date_list, hvac_pred=hvac_pred, prediction_data=prediction_data, dataframe=df_group_month.to_html(classes='dataframe'))

    except FileNotFoundError:
        return "File not found. Please check the file path."
    except Exception as e:
        return f"An error occurred: {str(e)}"
    

if __name__ == '__main__':
    app.run(debug=True)