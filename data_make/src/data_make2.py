#!/usr/bin/env python

import rospy, cv2, csv, time, os
import numpy as np
from tf.transformations import euler_from_quaternion
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image, LaserScan
from cv_bridge import CvBridge

os.chdir("/home/xytron/data/")

name = "data_make"
csv_name = "/home/xytron/data/data.csv"

cv_image = np.empty(shape=[0])
ranges = np.empty(shape=[0])

bridge = CvBridge()
current_p = [0, 0]
current_q = [0, 0, 0, 0]
scan_size = [500, 500]
resolution = 720
cnt = 0
edge = 240
f = open(csv_name, 'w')
wr = csv.writer(f)
wr.writerow(["ts_micro", "x", "y", "qx", "qy", "qz", "qw"])

if not os.path.isdir("/home/xytron/data/image/"):
    os.mkdir("/home/xytron/data/image/")
if not os.path.isdir("/home/xytron/data/image/background/"):
    os.mkdir("/home/xytron/data/image/background/")
if not os.path.isdir("/home/xytron/data/image/road/"):
    os.mkdir("/home/xytron/data/image/road/")
#if not os.path.isdir("/home/xytron/data/image/scan/"):
#    os.mkdir("/home/xytron/data/image/scan/")

def motor_callback(data):
    global current_p
    global current_q
    global wr
    i = data.header.stamp
    current_p = [data.pose.pose.position.x, data.pose.pose.position.y]
    current_q = [data.pose.pose.orientation.x, data.pose.pose.orientation.y, data.pose.pose.orientation.z, data.pose.pose.orientation.w]
    wr.writerow([str(i), current_p[0], current_p[1], current_q[0], current_q[1], current_q[2], current_q[3]])

def camera_callback(data):
    global cv_image
    global bridge, edge
    i = data.header.stamp
    cv_image = bridge.imgmsg_to_cv2(data, "bgr8")
    gray_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    
    background = gray_image[:edge,:]
    background = cv2.resize(background, dsize=(200, 112))
        
    road = gray_image[edge:,:]
    road = cv2.resize(road, dsize=(200, 112))
    
    #road = cv2.cvtColor(road, cv2.COLOR_BGR2GRAY)
    road = cv2.GaussianBlur(road,(5, 5), 0)
    road = cv2.Canny(np.uint8(road), 60, 70)

    cv2.imwrite("/home/xytron/data/image/background/"+str(i)+".jpg", background)
    cv2.imwrite("/home/xytron/data/image/road/"+str(i)+".jpg", road)

def lidar_callback(data):
    global resolution, scan_size
    ranges = data.ranges
    increment = data.angle_increment
    range_max = data.range_max
    i = data.header.stamp

    side = int(round(range_max * pow(2, 0.5)))
    half_side = float(side) / 2.0
    image_size = (side*100, side*100)
    img_space = np.full(image_size, 0, dtype=np.uint8)

    for i in range(0, resolution):
        if ranges[i] == float('inf'):
            continue
        radian = i * increment

        X = round((round(-ranges[i] * np.cos(radian), 4) + half_side) * 100)
        Y = round((round(ranges[i] * np.sin(radian), 4) + half_side) * 100)

        if (X >= image_size[0]) or (X < 0) or (Y >= image_size[1]) or (Y < 0):
            continue

        X = int(X)
        Y = int(Y)

        null_image[Y, X] = 255

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    dilate = cv2.dilate(null_image, kernel, iterations=10)
    ni = cv2.resize(dilate, (scan_size[0], scan_size[1]))
    cv2.imwrite("/home/xytron/data/image/scan/"+str(i)+".jpg", ni)

rospy.init_node(name, anonymous=True)
rospy.Subscriber("/carla/ego_vehicle/rgb_front/image", Image, camera_callback)
rospy.Subscriber("/carla/ego_vehicle/odometry", Odometry, motor_callback)
#rospy.Subscriber("/scan", LaserScan, lidar_callback)

rate = rospy.Rate(10)

while True:
    if current_p == [0, 0]:
        continue
    if cv_image.size != (640 * 480 * 3):
        continue
    print("start")
    break

while not rospy.is_shutdown():
    cv2.imshow("hehe", cv_image)
    cv2.waitKey(1)
    rate.sleep()

f.close()

