from flask import Flask, render_template, request, redirect, send_from_directory
import sqlite3
import os
import json
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
from datetime import datetime

app = Flask(__name__)

##########################################################
# Database Initialization
##########################################################

conn = sqlite3.connect("signup.db")
cur = conn.cursor()

##########################################################
# Users Table
##########################################################

cur.execute("""

CREATE TABLE IF NOT EXISTS users(

id INTEGER PRIMARY KEY AUTOINCREMENT,

name TEXT,

username TEXT,

email TEXT,

phone TEXT,

password TEXT

)

""")

##########################################################
# Prediction History Table
##########################################################

cur.execute("""

CREATE TABLE IF NOT EXISTS prediction_history(

id INTEGER PRIMARY KEY AUTOINCREMENT,

farmer TEXT,

district TEXT,

image_name TEXT,

analysis_image TEXT,

disease TEXT,

confidence REAL,

affected_percentage REAL,

severity TEXT,

prediction_date TEXT

)

""")

conn.commit()
conn.close()
##########################################################
# Upload Folder
##########################################################

UPLOAD_FOLDER = "uploads"

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

os.makedirs(

    UPLOAD_FOLDER,

    exist_ok=True

)

##########################################################
# Load CNN Model
##########################################################

model = load_model(

    "models/cnn.h5"

)

##########################################################
# Load Disease Classes
##########################################################

with open(

    "models/classes.json"

) as f:

    data = json.load(f)

classes = sorted(data, key=data.get)

##########################################################
# Image Preprocessing
##########################################################

def preprocess_image(path):

    img = cv2.imread(path)

    img = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2RGB
    )

    img = cv2.resize(
        img,
        (224,224)
    )

    img = np.array(img, dtype=np.float32)

    img = np.expand_dims(
        img,
        axis=0
    )

    img = preprocess_input(img)

    return img

##########################################################
# Estimate Visible Leaf Damage using OpenCV
##########################################################

def calculate_severity(image_path):

    img = cv2.imread(image_path)

    if img is None:
        return "Unknown", 0.0, None

    img = cv2.resize(img, (512,512))

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    ##################################################
    # Leaf Mask
    ##################################################

    lower_leaf = np.array([20,30,30])
    upper_leaf = np.array([95,255,255])

    leaf_mask = cv2.inRange(
        hsv,
        lower_leaf,
        upper_leaf
    )

    kernel = np.ones((5,5), np.uint8)

    leaf_mask = cv2.morphologyEx(
        leaf_mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    ##################################################
    # Diseased Region
    ##################################################

    h,s,v = cv2.split(hsv)

    healthy_green = (
        (h>25)&
        (h<95)&
        (s>40)
    )

    healthy_green = healthy_green.astype(np.uint8)*255

    disease_mask = cv2.bitwise_not(
        healthy_green
    )

    disease_mask = cv2.bitwise_and(
        disease_mask,
        leaf_mask
    )

    disease_mask = cv2.medianBlur(
        disease_mask,
        5
    )

    ##################################################
    # Find Diseased Spots
    ##################################################

    contours,_ = cv2.findContours(

        disease_mask,

        cv2.RETR_EXTERNAL,

        cv2.CHAIN_APPROX_SIMPLE

    )

    output = img.copy()

    diseased_area = 0

    for cnt in contours:

        area = cv2.contourArea(cnt)

        if area > 50:

            diseased_area += area

            cv2.drawContours(

                output,

                [cnt],

                -1,

                (0,0,255),

                2

            )

    total_leaf = cv2.countNonZero(
        leaf_mask
    )

    if total_leaf == 0:

        return "Healthy",0,None

    affected_percentage = round(

        (diseased_area/total_leaf)*100,

        2

    )

    if affected_percentage < 3:

        severity="Healthy"

    elif affected_percentage < 12:

        severity="Mild"

    elif affected_percentage < 30:

        severity="Moderate"

    else:

        severity="Severe"

    ##################################################
    # Save Result Image
    ##################################################

    analyzed_name = "analysis_" + os.path.basename(image_path)

    analyzed_path = os.path.join(

        app.config["UPLOAD_FOLDER"],

        analyzed_name

    )

    cv2.imwrite(

        analyzed_path,

        output

    )

    return severity, affected_percentage, analyzed_name
##########################################################
# Home
##########################################################

@app.route("/")

def home():

    return render_template(

        "home.html"

    )

##########################################################
# Signup
##########################################################

@app.route("/signup")

def signup():

    return render_template(

        "signup.html"

    )

##########################################################
# Save User
##########################################################

@app.route(

    "/save_user",

    methods=["POST"]

)

def save_user():

    name = request.form["name"]

    username = request.form["username"]

    email = request.form["email"]

    phone = request.form["phone"]

    password = request.form["password"]

    conn = sqlite3.connect(

        "signup.db"

    )

    cur = conn.cursor()

    cur.execute(

        """

        SELECT *

        FROM users

        WHERE email=?

        OR username=?

        """,

        (

            email,

            username

        )

    )

    existing = cur.fetchone()

    if existing:

        conn.close()

        return "User Already Exists"

    cur.execute(

        """

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

        """,

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

        "/signin_user"

    )

##########################################################
# User Login
##########################################################

@app.route("/signin_user")

def signin_user():

    return render_template(

        "signin_user.html"

    )

@app.route(

    "/user_login",

    methods=["POST"]

)

def user_login():

    username = request.form["username"]

    email = request.form["email"]

    password = request.form["password"]

    conn = sqlite3.connect(

        "signup.db"

    )

    cur = conn.cursor()

    cur.execute(

        """

        SELECT *

        FROM users

        WHERE username=?

        AND email=?

        AND password=?

        """,

        (

            username,

            email,

            password

        )

    )

    user = cur.fetchone()

    conn.close()

    if user:

        return redirect(

            "/user_dashboard"

        )

    else:

        return "Invalid Login"

##########################################################
# Admin Login
##########################################################

@app.route("/signin_admin")

def signin_admin():

    return render_template(

        "signin_admin.html"

    )

@app.route(

    "/admin_login",

    methods=["POST"]

)

def admin_login():

    username = request.form["username"]

    password = request.form["password"]

    if username=="admin" and password=="admin@123":

        return redirect(

            "/admin_dashboard"

        )

    else:

        return "Wrong Credentials"

##########################################################
# User Dashboard
##########################################################

@app.route("/user_dashboard")

def user_dashboard():

    return render_template(

        "user_dashboard.html"

    )
##########################################################
# Prediction
##########################################################

@app.route("/predict", methods=["POST"])
def predict():

    ##################################################
    # Get Form Data
    ##################################################

    farmer = request.form["farmer"]
    district = request.form["district"]
    image = request.files["image"]

    if image.filename == "":
        return "Please upload a crop image."

    ##################################################
    # Save Uploaded Image
    ##################################################

    filepath = os.path.join(
        app.config["UPLOAD_FOLDER"],
        image.filename
    )

    image.save(filepath)

    filename = image.filename


##################################################
# ResNet50 Prediction
##################################################

    img = preprocess_image(filepath)

    prediction = model.predict(img, verbose=0)

    index = np.argmax(prediction)

    disease = classes[index]

    disease = disease.replace("___", " - ")

    disease = disease.replace("_", " ")

    ##################################################
    # Prediction Confidence
    ##################################################

    confidence = float(round(np.max(prediction) * 100, 2))

    ##################################################
    # OpenCV Severity
    ##################################################

    severity, affected_percentage, analyzed_image = calculate_severity(
        filepath
    )
    ##################################################
# Recommendations Based on Affected Area
##################################################

    if "healthy" in disease.lower():

        affected_percentage = 0.0

        severity = "Healthy"

        suggestion = (
            "Crop appears healthy. Continue regular monitoring "
            "and follow good agricultural practices."
        )

        crop_tips = [

            "Monitor crop regularly.",

            "Maintain proper irrigation.",

            "Apply balanced fertilizers.",

            "Inspect leaves periodically.",

            "Maintain field hygiene."

        ]

        insurance = (
            "Need crop insurance assistance in the future? "
            "Click 'Explore PMFBY Resources' to learn about the scheme."
        )

        documents = []
        application_channels = []
        pmfby_links = []
        helpline = ""
        show_button = True


##################################################
# Early Stage (0–5%)
##################################################

    elif affected_percentage <= 5:

        severity = "Early Stage"

        suggestion = (
            f"Early symptoms of {disease} detected. Begin preventive "
            "treatment immediately to avoid further spread."
        )

        crop_tips = [

            f"Monitor {disease} symptoms every 2–3 days.",

            "Remove infected leaves carefully.",

            "Apply the recommended preventive fungicide or pesticide.",

            "Maintain proper field hygiene.",

            "Consult an Agriculture Officer if symptoms increase."

        ]

        insurance = (
            "Need crop insurance assistance in the future? "
            "Click 'Explore PMFBY Resources' to learn more."
        )

        documents=[]

        application_channels=[]

        pmfby_links=[]

        helpline=""

        show_button=True

##################################################
# Medium Stage (5–20%)
##################################################

    elif affected_percentage <= 20:

        severity = "Medium Stage"

        suggestion = (
            f"{disease} is spreading. Apply the recommended treatment "
            "and inspect the crop regularly."
        )

        crop_tips=[

            "Consult the nearest Agriculture Officer.",

            f"Apply disease-specific treatment for {disease}.",

            f"Inspect nearby plants for symptoms similar to {disease}.",

            "Improve field sanitation.",

            "Monitor disease progression regularly."

        ]

        insurance=(

            "Need crop insurance assistance in the future? "
            "Click 'Explore PMFBY Resources' to learn more."

        )

        documents=[]

        application_channels=[]

        pmfby_links=[]

        helpline=""

        show_button=True

    
##################################################
# High Stage (20–40%)
##################################################

    elif affected_percentage <= 40:

        severity = "High Stage"

        suggestion = (
            f"Significant infection of {disease} detected. "
            "Consult an Agriculture Officer and document the crop condition."
        )

        crop_tips=[

            f"Apply the recommended treatment for {disease} immediately.",

            "Capture photographs of affected leaves.",

            "Inspect surrounding plants carefully.",

            "Maintain cultivation records.",

            "Consult an Agriculture Officer."

        ]

        insurance=(

            "Need crop insurance assistance in the future? "
            "Click 'Explore PMFBY Resources' to learn more."

        )

        documents=[]

        application_channels=[]

        pmfby_links=[]

        helpline=""

        show_button=True
        
    ##################################################
    # Very High Stage (>40%)
    ##################################################

    else:

        severity = "Very High"

        suggestion = (
            f"Extensive visible damage caused by {disease} has been detected. "
            "Immediate expert intervention is strongly recommended."
        )

        crop_tips = [

            "Consult the nearest Agriculture Officer immediately.",

            "Capture photographs of the affected crop.",

            "Maintain cultivation records.",

            "Monitor the surrounding field."

        ]

        insurance = (
            "Extensive visible crop damage has been observed. "
            "Explore the official PMFBY resources below to learn "
            "about the scheme, required documents, application "
            "channels and farmer support services."
        )

        documents = [

            "Aadhaar Card",

            "Land Ownership Proof",

            "Bank Passbook",

            "Sowing Certificate (if applicable)",

            "Recent Crop Photographs"

        ]

        application_channels = [

            "PMFBY Official Portal",

            "Crop Insurance Mobile App",

            "Common Service Centre (CSC)",

            "Nearest Bank Branch",

            "Authorized Insurance Company"

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

        show_button = False

##################################################
# Save Prediction History
##################################################



    conn = sqlite3.connect("signup.db")
    cur = conn.cursor()

    cur.execute("""

    INSERT INTO prediction_history(

    farmer,
    district,
    image_name,
    analysis_image,
    disease,
    confidence,
    affected_percentage,
    severity,
    prediction_date

    )

    VALUES(?,?,?,?,?,?,?,?,?)

    """,

    (

    farmer,
    district,
    filename,
    analyzed_image,
    disease,
    confidence,
    affected_percentage,
    severity,
    datetime.now().strftime("%d-%m-%Y %H:%M")

    )

    )

    conn.commit()
    conn.close()
    
    ##################################################
    # Return Result Page
    ##################################################

    return render_template(

        "result.html",

        farmer=farmer,

        district=district,

        filename=filename,

        disease=disease,

        confidence=confidence,
        analysis_image=analyzed_image,

        severity=severity,

        affected_percentage=affected_percentage,

        suggestion=suggestion,

        crop_tips=crop_tips,

        insurance=insurance,

        documents=documents,

        application_channels=application_channels,

        pmfby_links=pmfby_links,

        helpline=helpline,

        show_button=show_button

    )
##########################################################
# Admin Dashboard
##########################################################

@app.route("/admin_dashboard")
def admin_dashboard():

    conn = sqlite3.connect("signup.db")
    cur = conn.cursor()

    cur.execute("SELECT COUNT(*) FROM users")

    total_users = cur.fetchone()[0]

    conn.close()

    return render_template(
        "admin_dashboard.html",
        total_users=total_users
    )

##########################################################
# Users
##########################################################

@app.route("/users")
def users():

    conn = sqlite3.connect("signup.db")
    cur = conn.cursor()

    cur.execute("SELECT * FROM users")

    data = cur.fetchall()

    conn.close()

    return render_template(
        "users.html",
        data=data
    )
##########################################################
# Prediction History
##########################################################

@app.route("/history")
def history():

    conn = sqlite3.connect("signup.db")
    cur = conn.cursor()

    cur.execute("""

    SELECT
    farmer,
    district,
    image_name,
    analysis_image,
    disease,
    confidence,
    affected_percentage,
    severity,
    prediction_date

    FROM prediction_history

    ORDER BY id DESC

    """)

    data = cur.fetchall()

    conn.close()

    return render_template(
        "history.html",
        data=data
    )
##########################################################
# Visualizations
##########################################################

@app.route("/visualizations")
def visualizations():

    return render_template(
        "visualizations.html"
    )

##########################################################
# Upload Train
##########################################################

@app.route("/upload_train")
def upload_train():

    return render_template(
        "upload_train.html"
    )

##########################################################
# Upload Test
##########################################################

@app.route("/upload_test")
def upload_test():

    return render_template(
        "upload_test.html"
    )

##########################################################
# Preprocess
##########################################################

@app.route("/preprocess")
def preprocess():

    message = """

Resize Completed

RGB Conversion Completed

Normalization Completed

"""

    return render_template(
        "preprocess.html",
        message=message
    )

##########################################################
# Train CNN
##########################################################

@app.route("/train_cnn")
def train_cnn():

    message = """

ResNet50 Model Trained Successfully

cnn.h5 Generated

classes.json Generated

Accuracy Graph Generated

Loss Graph Generated
"""

    return render_template(
        "train_result.html",
        message=message
    )

##########################################################
# Uploaded Images
##########################################################

@app.route("/uploads/<filename>")
def uploaded_file(filename):

    return send_from_directory(
        app.config["UPLOAD_FOLDER"],
        filename
    )

##########################################################
# Clear Prediction History
##########################################################

@app.route("/clear_history")
def clear_history():

    conn = sqlite3.connect("signup.db")
    cur = conn.cursor()

    cur.execute("DELETE FROM prediction_history")

    conn.commit()
    conn.close()

    return "Prediction history cleared successfully!"

##########################################################
# Run Flask App
##########################################################

if __name__ == "__main__":

    app.run(
        debug=True
    )