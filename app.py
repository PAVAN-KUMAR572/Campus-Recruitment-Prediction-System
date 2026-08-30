import numpy as np
from flask import Flask, request, render_template
import os
import pickle

app = Flask(__name__, template_folder="templates")

# Get the directory where the current script is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Paths to model files
placement_model_path = os.path.join(BASE_DIR, 'placement_model.pkl')
salary_model_path = os.path.join(BASE_DIR, 'salary_model.pkl')

# Global variables for models and scalers
placement_model = None
placement_scaler = None
salary_model = None
salary_scaler = None

# Helper function to safely convert inputs to float
def to_float(value, default=0.0):
    try:
        return float(value)
    except (ValueError, TypeError):
        return default

# Load models safely
def load_models():
    global placement_model, placement_scaler, salary_model, salary_scaler
    try:
        with open(placement_model_path, 'rb') as f:
            data = pickle.load(f)
            placement_model = data['model']
            placement_scaler = data['scaler']
        print("Placement model loaded successfully!")
    except Exception as e:
        print(f"Failed to load placement model: {e}")

    try:
        with open(salary_model_path, 'rb') as f:
            data = pickle.load(f)
            salary_model = data['model']
            salary_scaler = data['scaler']
        print("Salary model loaded successfully!")
    except Exception as e:
        print(f"Failed to load salary model: {e}")

load_models()

# Routes
@app.route('/')
def landing():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/index')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET'])
def predict():
    if placement_model is None or salary_model is None:
        return "Models are not loaded. Please check the server logs.", 500

    # Get inputs safely
    cgpa = to_float(request.args.get('cgpa', 0))
    projects = to_float(request.args.get('projects', 0))
    workshops = to_float(request.args.get('workshops', 0))
    mini_projects = to_float(request.args.get('mini_projects', 0))
    skills = request.args.get('skills', '')
    communication_skills = to_float(request.args.get('communication_skills', 0))
    internship = to_float(request.args.get('internship', 0))
    hackathon = to_float(request.args.get('hackathon', 0))
    tw_percentage = to_float(request.args.get('tw_percentage', 0))
    te_percentage = to_float(request.args.get('te_percentage', 0))
    backlogs = to_float(request.args.get('backlogs', 0))
    name = request.args.get('name', 'User')

    # Count skills
    skill_count = len([s for s in skills.split(',') if s.strip()])

    try:
        # Prepare input array (11 features)
        input_data = np.array([[cgpa, projects, workshops, mini_projects, skill_count,
                                communication_skills, internship, hackathon, tw_percentage,
                                te_percentage, backlogs]], dtype=float)

        # 1. Placement Prediction
        input_scaled_placement = placement_scaler.transform(input_data)
        output_raw = placement_model.predict(input_scaled_placement)[0]
        
        placed = str(output_raw).lower() == 'placed'

        # 2. Salary Prediction (using same 11 features)
        input_scaled_salary = salary_scaler.transform(input_data)
        salary_raw = salary_model.predict(input_scaled_salary)[0]
        
        salary_value = abs(int(round(salary_raw)))
        formatted_salary = f"{salary_value:,}"

        # Output messages
        if placed:
            out = f'Congratulations {name}!! You have high chances of getting placed!!!'
            out2 = f'Your expected salary will be INR {formatted_salary} per annum'
        else:
            out = f'Sorry {name}!! You have low chances of getting placed. All the best!!!!'
            out2 = 'Improve your skills to boost your placement chances.'

    except Exception as e:
        out = f"Error during prediction: {str(e)}"
        out2 = ''

    return render_template('out.html', output=out, output2=out2)

import webbrowser
from threading import Timer

def open_browser():
    webbrowser.open_new("http://127.0.0.1:5001")

if __name__ == "__main__":
    # The timer ensures the server has a second to start before the browser opens
    Timer(1, open_browser).start()
    app.run(debug=True, port=5001, use_reloader=False)



