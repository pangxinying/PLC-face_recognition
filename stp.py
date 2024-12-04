#!/usr/bin/env python3

import rospy
from std_msgs.msg import String
import os
import cv2
import face_recognition
import numpy as np
import time

def calculate_distance(face_location, frame_width):
    """
    使用人脸框的宽度来估算人与摄像头的距离
    假设摄像头的视角是已知的，并根据面部框的宽度来计算距离
    """
    known_face_width = 14.0  # 假设人脸宽度（单位：厘米）
    focal_length = 800  # 假设焦距（像素，通常根据相机进行标定）

    # 计算人脸框的宽度
    face_width_in_frame = face_location[2] - face_location[0]  # 右边界 - 左边界

    # 估算距离
    distance = (known_face_width * focal_length) / face_width_in_frame
    return distance

def face_recognition_publisher():
    # 初始化 ROS 节点
    rospy.init_node('face_recognition_node', anonymous=True)
    pub = rospy.Publisher('/face_recognition_node', String, queue_size=10)

    # 加载已知人脸编码（初始训练）
    known_faces = []
    known_face_names = []

    # 加载已知人脸数据
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

    # 打开摄像头
    video_capture = cv2.VideoCapture(0)

    if not video_capture.isOpened():
        rospy.logwarn("Failed to open camera!")
        return

    last_publish_time = time.time()  # 用于控制发布的时间间隔

    while not rospy.is_shutdown():
        ret, frame = video_capture.read()

        if not ret:
            rospy.logwarn("Failed to capture frame from camera!")
            continue

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 检测人脸
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        # 识别每个面部
        for face_encoding, face_location in zip(face_encodings, face_locations):
            # 计算人与摄像头的距离
            distance = calculate_distance(face_location, frame.shape[1])

            # 获取照片的名字
            name = "Unknown"
            face_distances = face_recognition.face_distance(known_faces, face_encoding)
            if len(face_distances) > 0:
                best_match_index = np.argmin(face_distances)
                if face_distances[best_match_index] < 0.4:
                    name = known_face_names[best_match_index]

            # 每 4 秒发布一次消息
            current_time = time.time()
            if current_time - last_publish_time >= 4:
                if distance < 100 and name != "Unknown":
                    rospy.loginfo("Face recognized within acceptable range")
                    pub.publish("granted")
                else:
                    rospy.loginfo("Face too far from camera, denied")
                    pub.publish("denied")  # 发布 denied 消息
                
                last_publish_time = current_time 

            top, right, bottom, left = face_location
            if name == "Unknown":
             # 未知人脸，框颜色为红色
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                cv2.putText(frame, "Unknown", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            else:
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

           
        # 显示图像
        cv2.imshow("Video", frame)

        # 按 'r' 键注册新的人脸，按 'q' 退出
        key = cv2.waitKey(1) & 0xFF
        if key == ord('r'):
            if len(face_encodings) > 0:
                new_face_encoding = face_encodings[0]
                # 保存新的人脸编码
                new_face_name = f"face_{len(known_faces) + 1}"
                known_faces.append(new_face_encoding)
                known_face_names.append(new_face_name)

                # 保存当前帧作为图片
                cv2.imwrite(f"{face_data_dir}/{new_face_name}.jpg", frame)
                rospy.loginfo(f"New face {new_face_name} added!")
                pub.publish("denied")  # Deny access while registering the face

        if key == ord('q'):  # Quit the program
            break

    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        face_recognition_publisher()
    except rospy.ROSInterruptException:
        pass
