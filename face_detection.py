import cv2 as cv

# Load the Haar Cascade classifier for face detection
face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')

# Capture video from the default camera
cap = cv.VideoCapture(0)