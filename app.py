from flask import Flask, request, render_template, redirect, url_for, flash, jsonify, Response, session
from flask_sqlalchemy import SQLAlchemy
import joblib
import os
import instaloader
from werkzeug.security import generate_password_hash, check_password_hash
from flask import render_template
import matplotlib.pyplot as plt
from io import StringIO, BytesIO
import base64
from datetime import datetime
import plotly.express as px
from plotly.offline import plot
import plotly.graph_objs as go
import cv2
import face_recognition
import numpy as np
import threading


app = Flask(__name__)
app.secret_key = 'your_secret_key_here'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///db1.db'
app.config['SQLALCHEMY_BINDS'] = {'db2': 'sqlite:///db2.db'}

db = SQLAlchemy(app)

# Create a list of known face encodings and their corresponding names
known_face_encodings = []
known_face_names = []

# Load known face images and create encodings for each known face
known_faces_dir = os.path.join(os.path.dirname(__file__), 'known_faces')
known_face_filenames = [
    "kartik.jpg",
    "atharva.jpeg",
    "dhruv.jpeg",
    "sai.jpeg",
    "rashika.jpeg",
]

for filename in known_face_filenames:
    image_path = os.path.join(known_faces_dir, filename)
    face_image = face_recognition.load_image_file(image_path)
    face_encoding = face_recognition.face_encodings(face_image)[0]  # Assuming one face per image
    known_face_encodings.append(face_encoding)
    known_face_names.append(filename.split(".")[0])


# Define the Flagged_User model
class Flagged_User(db.Model):
    __tablename__ = 'flagged_user'
    sr_flagged = db.Column(db.Integer, primary_key=True)
    username_flagged = db.Column(db.String(100), unique=True)
    count_flagged = db.Column(db.Integer, nullable=False)
    timestamp_flagged = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)

# Define the official_user model
# Define the Official_User model with a different column name for the password
class Official_User(db.Model):
    __tablename__ = 'official_user'
    sr_official = db.Column(db.Integer, primary_key=True)
    name_official = db.Column(db.String(100), nullable=False)
    email_official = db.Column(db.String(100), unique=True, nullable=False)
    hashed_password = db.Column(db.String(256), nullable=False)  # Change 'password' to 'hashed_password'


# Create database tables within the application context
with app.app_context():
    db.create_all()



@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']

        # Check if the email is already registered
        existing_user = Official_User.query.filter_by(email_official=email).first()
        if existing_user:
            flash('Email already registered. Please log in.')
            return redirect(url_for('login'))

        # Hash the password before storing it in the database
        hashed_password = generate_password_hash(password, method='scrypt')

        # Create a new official user and add it to the database
        new_user = Official_User(name_official=name, email_official=email, hashed_password=hashed_password)
        db.session.add(new_user)
        db.session.commit()

        flash('Registration successful. Please log in.')
        return redirect(url_for('login'))

    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        user = Official_User.query.filter_by(email_official=email).first()

        if user and check_password_hash(user.hashed_password, password):  # Use 'hashed_password' here
            # Log in the user
            # You can implement session management here
            flash('Login successful. Welcome!')
            return redirect(url_for('webcam_access', user_data = user))
        else:
            flash('Login failed. Please check your email and password.')

    return render_template('login.html')

# Import necessary libraries
import matplotlib.dates as mdates

@app.route('/webcam_access', methods=['GET', 'POST'])
def webcam_access():
    user_data = request.args.get('user_data')
    if user_data:
        return render_template('webcam_access.html', user_data=user_data)
    else:
        return redirect(url_for('login'))
    
@app.route('/dashboard', methods=['GET','POST'])
def dashboard():
    if session.get('authenticated'):
        selected_username = request.args.get('username')

        if selected_username:
            # Fetch and display details for the selected username
            instaloader_data = scrape_instagram_profile(selected_username)
            if instaloader_data:
                return render_template('dashboard.html', data=instaloader_data)
            else:
                # Handle the case where the Instagram profile does not exist
                return render_template('profile_not_found.html')

        # Query the database to fetch data for the pie chart
        flagged_users = Flagged_User.query.all()
        labels = [user.username_flagged for user in flagged_users]
        counts = [user.count_flagged for user in flagged_users]

        # Create a pie chart
        plt.figure(figsize=(8, 8))
        plt.pie(counts, labels=labels, autopct='%1.1f%%', startangle=140)
        plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

        # Save the pie chart as an image
        pie_chart_image_path = 'static/pie_chart.png'
        plt.savefig(pie_chart_image_path, bbox_inches='tight', pad_inches=0.1)

        # Fetch data for the bar chart (date and time of flagged users)
        timestamps = [user.timestamp_flagged for user in flagged_users]
        flag_counts = [user.count_flagged for user in flagged_users]

        # Create a Plotly bar chart
        bar_chart = go.Bar(
            x=timestamps,
            y=flag_counts,
            text=flag_counts,
            textposition='auto',
            marker=dict(color='blue'),  # You can customize the bar color
        )

        layout = go.Layout(
            title='Flagged Users Over Time',
            xaxis=dict(title='Timestamp'),
            yaxis=dict(title='Flag Counts'),
        )

        fig = go.Figure(data=[bar_chart], layout=layout)

        # Save the Plotly bar chart as an HTML file
        bar_chart_html_path = 'static/bar_chart.html'
        plot(fig, filename=bar_chart_html_path, auto_open=False)

        # username = 'some_username'  # Replace with the username you want to fetch data for
        # instaloader_data = scrape_instagram_profile(username)
        empty_data = {}
        # Pass the flagged user data and chart HTML path to the template
        return render_template('dashboard.html', pie_chart_image_path='static/pie_chart.png', bar_chart_html_path=bar_chart_html_path, flagged_users=flagged_users,data=empty_data)
    else:
        return render_template('login.html')
# Initialize OpenCV's Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

known_face_detected = False
video_thread = None  # Store the video thread globally

def face_detection_thread():
    global known_face_detected

    video_capture = cv2.VideoCapture(0)

    while not known_face_detected:
        ret, frame = video_capture.read()

        if not ret:
            # If the frame is not valid, skip the rest of the loop and continue capturing
            print("Invalid frame received.")
            continue

        # Convert the frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame using Haar Cascade
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            # Extract the face region from the frame
            face_image = frame[y:y+h, x:x+w]

            # Convert the face image to RGB format (required by face_recognition)
            rgb_face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)

            # Encode the face using face_recognition
            face_encodings = face_recognition.face_encodings(rgb_face_image)

            # Check if any known face matches the detected face
            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"

                if True in matches:
                    first_match_index = matches.index(True)
                    name = known_face_names[first_match_index]
                    known_face_detected = True  # Set the global flag to True
                    break  # Exit the loop when a known face is detected

        if known_face_detected:
            break

    video_capture.release()

def generate_frames():
    video_capture = cv2.VideoCapture(0)

    global known_face_detected  # Access the global flag

    while True:
        ret, frame = video_capture.read()

        # Check if the frame is valid
        if not ret or frame is None:
            continue  # Skip this iteration if the frame is invalid

        # Convert the frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame using Haar Cascade
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            # Extract the face region from the frame
            face_image = frame[y:y+h, x:x+w]

            # Convert the face image to RGB format (required by face_recognition)
            rgb_face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)

            # Encode the face using face_recognition
            face_encodings = face_recognition.face_encodings(rgb_face_image)

            # Check if any known face matches the detected face
            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"

                if True in matches:
                    first_match_index = matches.index(True)
                    name = known_face_names[first_match_index]
                    known_face_detected = True  # Set the global flag to True

                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (x + 6, y - 6), font, 0.5, (255, 255, 255), 1)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        if known_face_detected:
            break  # If a known face is detected, exit the loop

    # If a known face was detected, redirect to the dashboard outside the loop

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Route to check for known face and redirect to dashboard
@app.route('/check_known_face', methods=['POST'])
def check_known_face():
    global known_face_detected, video_thread

    if request.method == 'POST':
        # If the request is a POST (i.e., user clicked the "Authenticate" button)
        if video_thread is None or not video_thread.is_alive():
            # Start the face detection thread
            video_thread = threading.Thread(target=face_detection_thread)
            video_thread.start()

        # Wait for the face detection thread to finish
        video_thread.join()

        # Check if a known face is detected
        if known_face_detected:
            # Reset the flag for future requests
            known_face_detected = False
            # Set the session variable to indicate successful authentication
            session['authenticated'] = True
            # Redirect the user to the dashboard
            return redirect(url_for('dashboard'))
        else:
            # If no known face is detected, return a message indicating authentication failed
            return "Authentication failed. No known face detected."
    else:
        # If the request is a GET, the user is accessing the page initially
        return render_template('webcam_access.html')

@app.route('/fetch_instaloader_data/<username>')
def fetch_instaloader_data(username):
    instaloader_data = scrape_instagram_profile(username)
    
    if instaloader_data:
        # Return the user details as JSON
        return jsonify(instaloader_data)
    else:
        # Handle the case where the Instagram profile does not exist
        return jsonify({'error': 'Profile not found'}), 404

@app.route('/user_details/<username>')
def user_details(username):
    # Fetch user details for the given username (e.g., from the database)
    # Replace this with your actual data retrieval logic
    user = get_user_details(username)  # Implement this function

    # Render a template to display the user details
    return render_template('user_details.html', user=user)


@app.route('/flag_instagram_account/<username>', methods=['POST'])
def flag_instagram_account(username):
    if request.method == 'POST':
        # Check if the username is already flagged
        existing_flag = Flagged_User.query.filter_by(username_flagged=username).first()
        if existing_flag:
            # If the username is already flagged, increment the count
            existing_flag.count_flagged += 1
        else:
            # If it's not flagged, create a new entry with a count of 1
            new_flag = Flagged_User(username_flagged=username, count_flagged=1)
            db.session.add(new_flag)

        db.session.commit()

        flash('Instagram account flagged successfully.')
        return redirect(url_for('index'))

    # Handle GET requests to this route as needed
    return redirect(url_for('index'))


# Determine the directory of the current script
current_directory = os.path.dirname(os.path.abspath(__file__))

# Define the directory path where the model is stored relative to the script's directory
model_dir = os.path.join(current_directory, 'models')
model_filename = 'xgboost_model.pkl'

# Get the absolute path to the model
model_path = os.path.join(model_dir, model_filename)

# Load the XGBoost model
xgb_classifier = joblib.load(model_path)

# Function to scrape Instagram profile data and select relevant attributes for display
def scrape_instagram_profile(username):
    loader = instaloader.Instaloader()
    try:
        profile = instaloader.Profile.from_username(loader.context, username)
        
        # Extract the desired attributes for display
        instagram_data = {
            "Username": username,
            "Full Name": profile.full_name,
            "Biography": profile.biography,
            "Followers Count": profile.followers,
            "Following Count": profile.followees,
            "Has Profile Picture": int(bool(profile.profile_pic_url)),
            "Numerical Chars in Username": sum(c.isdigit() for c in username) / len(username),
            "Username Length": len(username),
            "Ratio Numerical Chars in Username": sum(c.isdigit() for c in username) / len(username),
            "Full Name Tokens": len(profile.full_name.split()),
            "Numerical Chars in Full Name": sum(c.isdigit() for c in profile.full_name) / len(profile.full_name),
            "Full Name Length": len(profile.full_name),
            "Ratio Numerical Chars in Full Name": sum(c.isdigit() for c in profile.full_name) / len(profile.full_name),
            "Same Username and Full Name": int(username.lower() == profile.full_name.lower()),
            "Description Length": len(profile.biography),
            "External URL": int(bool(profile.external_url)),
            "Private": int(profile.is_private),
            "Number of Posts": profile.mediacount,
        }

        return instagram_data
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
            conclusion = "Not Fake"
        else:
            conclusion = "Fake"

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
            input_data = [[
                scraped_data["Has Profile Picture"],
                scraped_data["Numerical Chars in Username"],
                scraped_data["Full Name Tokens"],
                scraped_data["Numerical Chars in Full Name"],
                scraped_data["Same Username and Full Name"],
                scraped_data["Description Length"],
                scraped_data["External URL"],
                scraped_data["Private"],
                scraped_data["Number of Posts"],
                scraped_data["Followers Count"],
                scraped_data["Following Count"],
            ]]

            # Make predictions using the model
            prediction = xgb_classifier.predict(input_data)

            # Convert the prediction to a human-readable conclusion
            if prediction == 0:
                conclusion = "Not Fake"
            else:
                conclusion = "Fake"

            # Render the result template with the prediction and fetched data
            return render_template('result_instagram.html', prediction=conclusion, data=scraped_data)
        else:
            # Handle the case where the Instagram profile does not exist
            return render_template('profile_not_found.html')

if __name__ == '__main__':
    app.run(debug=True)
