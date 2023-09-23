from flask import Flask, request, render_template, jsonify
import joblib
import os
import instaloader

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

# Function to scrape Instagram profile data
def scrape_instagram_profile(username):
    loader = instaloader.Instaloader()
    try:
        profile = instaloader.Profile.from_username(loader.context, username)
        has_profile_picture = int(bool(profile.profile_pic_url))
        numerical_chars_in_username = sum(c.isdigit() for c in username)
        username_length = len(username)
        ratio_numerical_chars_in_username = numerical_chars_in_username / username_length
        full_name_tokens = len(profile.full_name.split())
        numerical_chars_in_full_name = sum(c.isdigit() for c in profile.full_name)
        full_name_length = len(profile.full_name)
        ratio_numerical_chars_in_full_name = numerical_chars_in_full_name / full_name_length
        same_username_and_full_name = int(username.lower() == profile.full_name.lower())
        bio_length = len(profile.biography)
        has_external_url = int(bool(profile.external_url))
        is_private = int(profile.is_private)
        num_posts = profile.mediacount

        return [has_profile_picture, ratio_numerical_chars_in_username, full_name_tokens,
                ratio_numerical_chars_in_full_name, same_username_and_full_name, bio_length,
                has_external_url, is_private, num_posts, profile.followers, profile.followees]
    except instaloader.exceptions.ProfileNotExistsException:
        return None

@app.route('/')
def index():
    return render_template('index.html')

# Route for Option 1: Input form with manual input
@app.route('/input_manual_form')
def input_manual_form():
    # Render the HTML form for manual input
    return render_template('input_manual_form.html')

# Route for Option 1: Predict with manual input
@app.route('/predict_manual', methods=['POST'])
def predict_manual():
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
            conclusion = "Not Spam (Option 1)"
        else:
            conclusion = "Spam (Option 1)"

        # Render the result template with the prediction
        return render_template('result.html', prediction=conclusion)

# Route for Option 2: Input form for Instagram username
@app.route('/input_username_form')
def input_username_form():
    # Render the HTML form for entering the Instagram username
    return render_template('input_username_form.html')

# Route for Option 2: Predict with Instagram username input
@app.route('/predict_instagram_data', methods=['POST'])
def predict_instagram_data():
    if request.method == 'POST':
        # Extract Instagram username from the form
        username = request.form['username']

        # Scrape Instagram profile data
        scraped_data = scrape_instagram_profile(username)

        if scraped_data:
            # Prepare the input data for prediction
            input_data = [scraped_data]

            # Make predictions using the model
            prediction = xgb_classifier.predict(input_data)

            # Convert the prediction to a human-readable conclusion
            if prediction == 0:
                conclusion = "Not Spam (Option 2)"
            else:
                conclusion = "Spam (Option 2)"

            # Render the result template with the prediction
            return render_template('result.html', prediction=conclusion)
        else:
            # Handle the case where the Instagram profile does not exist
            return render_template('profile_not_found.html')

if __name__ == '__main__':
    app.run(debug=True)
