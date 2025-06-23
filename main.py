import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

# Define the path to the directory containing known face images
path = 'images' 

images = []
classNames = []

# Get a list of all file names in the 'images' directory
myList = os.listdir(path)

for cls in myList:
    curImg = cv2.imread(f'{path}/{cls}')  
    images.append(curImg) 
    classNames.append(os.path.splitext(cls)[0]) 

# Function to find face encodings for a list of images
def findEncodings(images):
    encodeList = []
    for img in images:
        # Convert the image from BGR (OpenCV default) to RGB (face_recognition requirement)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0] # Get face encodings for the current image.
        encodeList.append(encode)
    return encodeList

# Function to mark attendance for a recognized person

def markAttendance(name):
    # Create directory if it doesn't exist
    folder = 'Attendance_Records'
    os.makedirs(folder, exist_ok=True)

    # Create filename based on today's date inside the folder
    date_str = datetime.now().strftime('%d-%m-%Y')
    filename = os.path.join(folder, f"{date_str}.csv")

    # If file doesn't exist, create and write header
    if not os.path.isfile(filename):
        with open(filename, 'w', newline='') as f:
            f.write('Name,Time\n')

    # Read existing names from the file
    with open(filename, 'r+') as f:
        lines = f.readlines()
        names = [line.split(',')[0] for line in lines[1:]]  # Skip header

        # If name not recorded, append it with date and time
        if name not in names:
            time_str = datetime.now().strftime('%H:%M:%S')
            f.write(f'{name},{time_str}\n')

# Generate encodings for all known faces
encodeListKnowm = findEncodings(images)
print("Encoding Completes....")

# Initialize the webcam (0 for default camera)
cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    imgS = cv2.resize(img,(0,0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    faceCurFrame = face_recognition.face_locations(imgS)
    encodeCurFrame = face_recognition.face_encodings(imgS, faceCurFrame)

    for encodeFace, faceLoc in zip(encodeCurFrame, faceCurFrame):
        matches = face_recognition.compare_faces(encodeListKnowm, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnowm, encodeFace)
        matchInd = np.argmin(faceDis)

        # Check if a match is found and the face distance is below a threshold (0.5 typically)
        if matches[matchInd] and faceDis[matchInd] < 0.5:
            name = classNames[matchInd].upper()
            t, r, l, b = faceLoc
            t, r, l, b = t*4, r*4, l*4, b*4
            cv2.rectangle(img, (b,t),(r,l), (0,255,0),2)
            cv2.rectangle(img,(b, l-35),(r,l),(0,255,0),cv2.FILLED)
            cv2.putText(img, name, (b+6, l-6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255),2)
            markAttendance(name)

    cv2.imshow('Face Attendance',img)
    # If 'Enter' is pressed, break out of the loop
    if cv2.waitKey(1) == 13:
        break

cap.release()
cv2.destroyAllWindows()
