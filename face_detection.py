import cv2 as cv
import streamlit as st
import numpy as np

# Load the Haar Cascade classifier for face detection
face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
# print("hiii")
# Set up Streamlit page configuration
st.title("Face Detection App")
st.text("This app detects faces in real-time from your webcam.")

# Initialize Streamlit session state to control start and stop actions
if "run" not in st.session_state:
    st.session_state["run"] = False

# Define start and stop buttons
start_button = st.button("Start Face Detection")
stop_button = st.button("Stop Face Detection")

# Start button toggles the session state to True
if start_button:
    st.session_state["run"] = True

# Stop button toggles the session state to False
if stop_button:
    st.session_state["run"] = False

# Video capture loop
if st.session_state["run"]:
    # Capture video from the default camera
    cap = cv.VideoCapture(0)
    frame_placeholder = st.empty()

    while st.session_state["run"]:
        # Read a frame from the video capture
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to grayscale
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # Detect faces in the grayscale frame
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        # Draw rectangles around the detected faces
        for (x, y, w, h) in faces:
            cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Convert the frame color to RGB for Streamlit display
        frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

        # Display the frame in Streamlit
        frame_placeholder.image(frame_rgb, channels="RGB")

    # Release the video capture when stopping
    cap.release()
