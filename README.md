# Real-Time-Face-Recognition-Attendance-System

Description:

This project is a Real-Time Face Recognition Attendance System developed using Python, OpenCV, and the face_recognition library. It captures faces from a live webcam feed, matches them with pre-stored images, and automatically logs attendance into a CSV file with timestamps.

Key Features:

Automatically loads and encodes known faces from a specified image directory.
Scales input video frames to enhance performance and reduce processing load.
Detects and recognizes faces in real time, displaying names with bounding boxes.
Logs attendance to a date-stamped .csv file, ensuring no duplicate entries per day.
Includes false match filtering using face distance metrics for improved accuracy.

Technologies Used:

Python, OpenCV, face_recognition, NumPy, CSV, datetime, OS
