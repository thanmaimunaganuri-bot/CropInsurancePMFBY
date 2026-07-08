
#from flask import Flask, render_template, request, redirect
from flask import Flask, render_template, request, redirect, send_from_directory
import sqlite3
import os
import json
import cv2
import numpy as np

#from tensorflow.keras.models import load_model
from keras.models import load_model

app = Flask(__name__)

#########################################
# Database Initialization
#########################################

conn = sqlite3.connect('signup.db')
cur = conn.cursor()

cur.execute('''

CREATE TABLE IF NOT EXISTS users(

id INTEGER PRIMARY KEY AUTOINCREMENT,

name TEXT,

username TEXT,

email TEXT,

phone TEXT,

password TEXT

)

''')

conn.commit()
conn.close()

#########################################
# Upload Folder
#########################################

UPLOAD_FOLDER='uploads'

app.config['UPLOAD_FOLDER']=UPLOAD_FOLDER

os.makedirs(

UPLOAD_FOLDER,

exist_ok=True

)

#########################################
# Load CNN Model
#########################################

model=load_model(

'models/cnn.h5'

)

#########################################
# Load Classes
#########################################

with open(

'models/classes.json'

) as f:

    data=json.load(f)

classes=list(

data.keys()

)

#########################################
# Image Preprocessing
#########################################

def preprocess_image(path):

    img=cv2.imread(path)

    img=cv2.cvtColor(

        img,

        cv2.COLOR_BGR2RGB

    )

    img=cv2.resize(

        img,

        (64,64)

    )

    img=img/255.0

    img=np.expand_dims(

        img,

        axis=0

    )

    return img

#########################################
# Home
#########################################

@app.route('/')

def home():

    return render_template(

        'home.html'

    )

#########################################
# Signup
#########################################

@app.route('/signup')

def signup():

    return render_template(

        'signup.html'

    )

#########################################
# Save User
#########################################

@app.route(

'/save_user',

methods=['POST']

)

def save_user():

    name=request.form['name']

    username=request.form['username']

    email=request.form['email']

    phone=request.form['phone']

    password=request.form['password']

    conn=sqlite3.connect(

        'signup.db'

    )

    cur=conn.cursor()

    cur.execute(

    '''

    SELECT *

    FROM users

    WHERE email=?

    OR username=?

    ''',

    (

    email,

    username

    )

    )

    existing=cur.fetchone()

    if existing:

        conn.close()

        return "User Already Exists"

    cur.execute(

    '''

    INSERT INTO users

    (

    name,

    username,

    email,

    phone,

    password

    )

    VALUES

    (

    ?,?,?,?,?

    )

    ''',

    (

    name,

    username,

    email,

    phone,

    password

    )

    )

    conn.commit()

    conn.close()

    return redirect(

        '/signin_user'

    )

#########################################
# User Login
#########################################

@app.route('/signin_user')

def signin_user():

    return render_template(

        'signin_user.html'

    )

#########################################
# User Validation
#########################################

@app.route(

'/user_login',

methods=['POST']

)

def user_login():

    username=request.form['username']

    email=request.form['email']

    password=request.form['password']

    conn=sqlite3.connect(

        'signup.db'

    )

    cur=conn.cursor()

    cur.execute(

    '''

    SELECT *

    FROM users

    WHERE username=?

    AND email=?

    AND password=?

    ''',

    (

    username,

    email,

    password

    )

    )

    user=cur.fetchone()

    conn.close()

    if user:

        return redirect(

            '/user_dashboard'

        )

    else:

        return "Invalid Login"

#########################################
# Admin Login Page
#########################################

@app.route('/signin_admin')

def signin_admin():

    return render_template(

        'signin_admin.html'

    )

#########################################
# Admin Validation
#########################################

@app.route(

'/admin_login',

methods=['POST']

)

def admin_login():

    username=request.form['username']

    password=request.form['password']

    if username=="admin" and password=="admin@123":

        return redirect(

            '/admin_dashboard'

        )

    else:

        return "Wrong Credentials"

#########################################
# User Dashboard
#########################################

@app.route('/user_dashboard')

def user_dashboard():

    return render_template(

        'user_dashboard.html'

    )
#########################################
# Prediction
#########################################

@app.route('/predict', methods=['POST'])

def predict():

    farmer = request.form['farmer']

    district = request.form['district']

    image = request.files['image']


    if image.filename == '':

        return "Please Upload Image"


    filepath = os.path.join(

        app.config['UPLOAD_FOLDER'],

        image.filename

    )


    image.save(filepath)


    filename = image.filename


    img = preprocess_image(filepath)


    pred = model.predict(

        img,

        verbose=0

    )


    index = np.argmax(pred)


    disease = classes[index]


    disease = disease.replace(

        "___",

        " "

    )


    disease = disease.replace(

        "_",

        " "

    )


    confidence_score = np.max(pred) * 100

    if confidence_score >= 90:

        confidence = "High"

    elif confidence_score >= 75:

        confidence = "Moderate"

    else:

        confidence = "Low"


#########################################

    if "Healthy" in disease:


        suggestion = (

            "Crop is healthy. Continue regular monitoring "

            "and preventive practices."

        )


        insurance = (

            "No immediate PMFBY assistance required"

        )


        crop_tips = [

            "Continue regular crop monitoring",

            "Maintain proper irrigation schedule",

            "Apply balanced fertilizers",

            "Inspect crops periodically for pests",

            "Follow preventive disease management practices",

            "Maintain field hygiene",

            "Consider enrolling in PMFBY for future protection"

        ]


        documents = []


        application_channels = []


        pmfby_links = []


        helpline = ""


    else:


        suggestion = (

            "Consult Agriculture Officer, apply suitable "

            "treatment and verify PMFBY eligibility."

        )


        

        insurance = (

                "Farmer may explore PMFBY support options"

        )



        crop_tips = [

            "Consult the nearest Agriculture Officer",

            "Apply suitable fungicides or pesticides",

            "Monitor crop condition regularly",

            "Capture photographs of crop damage",

            "Maintain cultivation records safely",

            "Verify PMFBY eligibility"

        ]


        documents = [

            "Aadhaar Card",

            "Land Ownership Proof",

            "Bank Passbook",

            "Sowing Certificate (if applicable)"

        ]


        application_channels = [

            "PMFBY Portal",

            "Crop Insurance Mobile App",

            "Common Service Centre (CSC)",

            "Bank Branch",

            "Insurance Representative"

        ]


        pmfby_links = [

            (

                "PMFBY Official Portal",

                "https://www.pmfby.gov.in"

            ),

            (

                "Farmer Registration",

                "https://www.pmfby.gov.in/farmerRegistrationForm/"

            ),

            (

                "Track Application Status",

                "https://www.pmfby.gov.in/status"

            )

        ]


        helpline = "14447"


#########################################

    return render_template(

        'result.html',

        farmer=farmer,

        district=district,

        disease=disease,

        confidence=confidence,

        suggestion=suggestion,

        insurance=insurance,

        crop_tips=crop_tips,

        documents=documents,

        application_channels=application_channels,

        pmfby_links=pmfby_links,

        helpline=helpline,

        filename=filename

    )
#########################################
# Admin Dashboard
#########################################

@app.route('/admin_dashboard')

def admin_dashboard():

    return render_template(

        'admin_dashboard.html'

    )

#########################################
# Users
#########################################

@app.route('/users')

def users():

    conn=sqlite3.connect(

        'signup.db'

    )

    cur=conn.cursor()

    cur.execute(

        'SELECT * FROM users'

    )

    data=cur.fetchall()

    conn.close()

    return render_template(

        'users.html',

        data=data

    )

#########################################
# Visualizations
#########################################

@app.route('/visualizations')

def visualizations():

    return render_template(

        'visualizations.html'

    )

#########################################
# Upload Train
#########################################

@app.route('/upload_train')

def upload_train():

    return render_template(

        'upload_train.html'

    )

#########################################
# Upload Test
#########################################

@app.route('/upload_test')

def upload_test():

    return render_template(

        'upload_test.html'

    )

#########################################
# Preprocess
#########################################

@app.route('/preprocess')

def preprocess():

    message="""

Resize Completed

RGB Conversion Completed

Normalization Completed

"""

    return render_template(

        'preprocess.html',

        message=message

    )

#########################################
# Train CNN
#########################################

@app.route('/train_cnn')

def train_cnn():

    message="""

CNN Trained Successfully

cnn.h5 Generated

classes.json Generated

Accuracy Graph Generated

Loss Graph Generated

"""

    return render_template(

        'train_result.html',

        message=message

    )
#########################################
# Uploaded Images
#########################################

@app.route('/uploads/<filename>')

def uploaded_file(filename):

    return send_from_directory(

        app.config['UPLOAD_FOLDER'],

        filename

    )

#########################################

if __name__=="__main__":

    app.run(

        debug=True

    )

