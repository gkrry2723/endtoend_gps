#!/usr/bin/env python

import rospy, cv2, csv, time
import numpy as np
from tf.transformations import euler_from_quaternion
from cv_bridge import CvBridge

from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image


name = "data_make"
video_name = "/home/xytron/data_made/carla_drive3.mp4"
csv_name = "/home/xytron/data_made/carla_drive3.csv"

cv_image = np.empty(shape=[0])
bridge = CvBridge()
current_p = [0, 0]
current_q = [0,0,0,0]
roll, pitch, yaw = 0,0,0

def motor_callback(data):
    global current_p
    global current_q
    global roll, pitch, yaw
    current_p = [data.pose.pose.position.x, data.pose.pose.position.y]
    current_q = [data.pose.pose.orientation.x, data.pose.pose.orientation.y, data.pose.pose.orientation.z, data.pose.pose.orientation.w]
    (roll, pitch, yaw) = euler_from_quaternion(current_q)
    print "yaw : ", yaw
    

def camera_callback(data):
    global cv_image
    global bridge
    cv_image = bridge.imgmsg_to_cv2(data, "bgr8")
    print(cv_image.shape)

rospy.init_node(name, anonymous=True)
rospy.Subscriber("/carla/ego_vehicle/rgb_front/image", Image, camera_callback)
rospy.Subscriber("/carla/ego_vehicle/odometry", Odometry, motor_callback)

out = cv2.VideoWriter(video_name,cv2.VideoWriter_fourcc('M','J','P','G'), 10, (800, 600))
f = open(csv_name, 'w')
wr = csv.writer(f)
wr.writerow(["ts_micro", "frame_index", "x", "y", "yaw"])

rate = rospy.Rate(10)
cnt = 0

while True:
    if current_p == [0, 0]:
        #print("in1")
        continue
    if cv_image.size != (800 * 600 * 3):
        #print("in2")
        continue
    print("start")
    break

while not rospy.is_shutdown():
    cv2.imshow("hehe", cv_image)
    wr.writerow([time.time(), cnt, current_p[0], current_p[1], yaw])
    out.write(cv_image)
    rate.sleep()
    
    #if cnt == 100:
    #    break
    cnt += 1

out.release()
f.close()

