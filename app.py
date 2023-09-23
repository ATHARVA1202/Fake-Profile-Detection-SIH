from flask import Flask, request, render_template
import joblib
import os

app = Flask(__name__)

# Determine the directory of the current script
current_directory = os.path.dirname(os.path.abspath(__file__))

# Define the directory path where the model is stored relative to the script's directory
model_dir = os.path.join(current_directory, 'models')
model_filename = 'xgboost_model.pkl'

# Get the absolute path to the model
model_path = os.path.join(model_dir, model_filename)

# Load the XGBoost model
xgb_classifier = joblib.load(model_path)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/input_form')
def input_form():
    # Render the HTML form for user input
    return render_template('input_form.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Extract input values from the form
        profile_pic = float(request.form['profile_pic'])
        nums_length_username = float(request.form['nums_length_username'])
        fullname_words = float(request.form['fullname_words'])
        nums_length_fullname = float(request.form['nums_length_fullname'])
        name_equals_username = float(request.form['name_equals_username'])
        description_length = float(request.form['description_length'])
        external_url = float(request.form['external_url'])
        private = float(request.form['private'])
        num_posts = float(request.form['num_posts'])
        num_followers = float(request.form['num_followers'])
        num_follows = float(request.form['num_follows'])

        # Prepare the input data for prediction
        input_data = [[profile_pic, nums_length_username, fullname_words, nums_length_fullname,
                       name_equals_username, description_length, external_url, private, num_posts,
                       num_followers, num_follows]]

        # Make predictions using the model
        prediction = xgb_classifier.predict(input_data)

        # Convert the prediction to a human-readable conclusion
        if prediction == 0:
            conclusion = "Not Spam"
        else:
            conclusion = "Spam"

        # Render the result template with the prediction
        return render_template('result.html', prediction=conclusion)

if __name__ == '__main__':
    app.run(debug=True)
