import cv2
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not access the camera.")
else:
    print("Camera is accessible.")
cap.release()
