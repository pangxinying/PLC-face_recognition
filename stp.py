#!/usr/bin/env python3

import rospy
from std_msgs.msg import String
import os
import cv2
import face_recognition
import numpy as np
import time

def calculate_distance(face_location, frame_width):

    known_face_width = 14.0  
    focal_length = 800 

    face_width_in_frame = face_location[2] - face_location[0] 


    distance = (known_face_width * focal_length) / face_width_in_frame
    return distance

def face_recognition_publisher():
    rospy.init_node('face_recognition_node', anonymous=True)
    pub = rospy.Publisher('/face_recognition_node', String, queue_size=10)

    known_faces = []
    known_face_names = []

    face_data_dir = "known_faces"
    if not os.path.exists(face_data_dir):
        os.makedirs(face_data_dir)
    else:
        for filename in os.listdir(face_data_dir):
            if filename.endswith(".jpg"):
                image = face_recognition.load_image_file(os.path.join(face_data_dir, filename))
                encodings = face_recognition.face_encodings(image)
                if len(encodings) > 0:
                    encoding = encodings[0]
                    known_faces.append(encoding)
                    known_face_names.append(filename.split(".")[0])

    video_capture = cv2.VideoCapture(0)

    if not video_capture.isOpened():
        rospy.logwarn("Failed to open camera!")
        return

    last_publish_time = time.time()  


    while not rospy.is_shutdown():
        ret, frame = video_capture.read()

        if not ret:
            rospy.logwarn("Failed to capture frame from camera!")
            continue

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for face_encoding, face_location in zip(face_encodings, face_locations):
            distance = calculate_distance(face_location, frame.shape[1])

            name = "Unknown"
            face_distances = face_recognition.face_distance(known_faces, face_encoding)
            if len(face_distances) > 0:
                best_match_index = np.argmin(face_distances)
                if face_distances[best_match_index] < 0.4:
                    name = known_face_names[best_match_index]

            current_time = time.time()
            if current_time - last_publish_time >= 4:
                if distance < 100 and name != "Unknown":
                    rospy.loginfo("Face recognized within acceptable range")
                    pub.publish("granted")
                else:
                    rospy.loginfo("Face too far from camera, denied")
                    pub.publish("denied")  

                
                last_publish_time = current_time 

            top, right, bottom, left = face_location
            if name == "Unknown":
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                cv2.putText(frame, "Unknown", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            else:
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

           
        cv2.imshow("Video", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('r'):
            if len(face_encodings) > 0:
                new_face_encoding = face_encodings[0]

                new_face_name = f"face_{len(known_faces) + 1}"
                known_faces.append(new_face_encoding)
                known_face_names.append(new_face_name)

                cv2.imwrite(f"{face_data_dir}/{new_face_name}.jpg", frame)
                rospy.loginfo(f"New face {new_face_name} added!")
                pub.publish("denied")  

        if key == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        face_recognition_publisher()
    except rospy.ROSInterruptException:
        pass
