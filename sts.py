#!/usr/bin/env python3

import rospy
from std_msgs.msg import String
from gtts import gTTS
import os

def face_recognition_callback(msg):
    # 收到消息时的回调函数
    rospy.loginfo(f"Received message: {msg.data}")
    
    if msg.data == "granted":
        rospy.loginfo("Access granted.")
        # 使用 gTTS 播放“granted”语音
        tts = gTTS(text='Access granted.Unlocking the door...', lang='en')
        tts.save("/tmp/granted.mp3")
        os.system("mpg321 /tmp/granted.mp3")  # 使用 mpg321 播放语音
    elif msg.data == "denied":
        rospy.loginfo("Access denied.Keeping the door locked...")
        # 使用 gTTS 播放“denied”语音
        tts = gTTS(text='Access denied.', lang='en')
        tts.save("/tmp/denied.mp3")
        os.system("mpg321 /tmp/denied.mp3")  # 使用 mpg321 播放语音
    else:
        rospy.logwarn(f"Unknown message received: {msg.data}")

def face_recognition_subscriber():
    rospy.init_node('face_recognition_subscriber_node', anonymous=True)
    
    # 订阅来自 publisher 的消息
    rospy.Subscriber('/face_recognition_node', String, face_recognition_callback)

    rospy.spin()

if __name__ == '__main__':
    try:
        face_recognition_subscriber()
    except rospy.ROSInterruptException:
        pass
