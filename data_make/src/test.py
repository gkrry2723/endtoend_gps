#!/usr/bin/env python

import rospy, cv2, csv, time, os
import numpy as np
from tf.transformations import euler_from_quaternion
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image#, LaserScan
from cv_bridge import CvBridge

os.chdir("/home/xytron/data/")

name = "data_make"
csv_name = "/home/xytron/data/data.csv"

camera_topic = "/carla/ego_vehicle/rgb_front/image"
odom_topic = "/carla/ego_vehicle/odometry"

camera_size_filter = (800 * 600 * 3)

cv_image = np.empty(shape=[0])
ranges = np.empty(shape=[0])

bridge = CvBridge()
current_p = [0, 0]
current_q = [0, 0, 0, 0]
#scan_size = [500, 500]
#resolution = 720
cnt = 0
edge = 240

camera_time = 0
lidar_time = 0
imu_time = 0
motor_time = 0

f = open(csv_name, 'w')
wr = csv.writer(f)
wr.writerow(["ts_micro", "x", "y", "qx", "qy", "qz", "qw"])

print(1)
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
    global motor_time
    motor_time = data.header.stamp
    current_p = [data.pose.pose.position.x, data.pose.pose.position.y]
    current_q = [data.pose.pose.orientation.x, data.pose.pose.orientation.y, data.pose.pose.orientation.z, data.pose.pose.orientation.w]

def camera_callback(data):
    global cv_image, camera_time
    camera_time = data.header.stamp
    cv_image = bridge.imgmsg_to_cv2(data, "bgr8")

#def lidar_callback(data):
#    global resolution, scan_size, lidar_time, ranges
#    ranges = data.ranges
#    increment = data.angle_increment
#    range_max = data.range_max
#    lidar_time = data.header.stamp

 
rospy.init_node(name, anonymous=True)
rospy.Subscriber(camera_topic, Image, camera_callback)
rospy.Subscriber(odom_topic, Odometry, motor_callback)
#rospy.Subscriber("/scan", LaserScan, lidar_callback)

rate = rospy.Rate(10)

while True:
    if current_p == [0, 0]:
        continue
    if cv_image.size != camera_size_filter:
        print("1",cv_image.size)
        print("2",camera_size_filter)
        continue
    print("start")
    break

#side = int(round(range_max * pow(2, 0.5)))
#half_side = float(side) / 2.0
#image_size = (side*100, side*100)
#img_space = np.full(image_size, 0, dtype=np.uint8)
cnt = 0
while not rospy.is_shutdown():
    #print("get in")
    cnt += 1
    #cv2.imshow("hehe", cv_image)
    cv_grayscale = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    background = cv_grayscale[:edge,:]
    background = cv2.resize(background, dsize=(200, 112))
        
    road = cv_grayscale[edge:,:]
    road = cv2.resize(road, dsize=(200, 112))
    #road = cv2.cvtColor(road, cv2.COLOR_BGR2GRAY)
    road = cv2.GaussianBlur(road,(5, 5), 0)
    road = cv2.Canny(np.uint8(road), 60, 70)

    #for i in range(0, resolution):
    #    if ranges[i] == float('inf'):
    #        continue
    #    radian = i * increment

    #    X = round((round(-ranges[i] * np.cos(radian), 4) + half_side) * 100)
    #    Y = round((round(ranges[i] * np.sin(radian), 4) + half_side) * 100)

    #    if (X >= image_size[0]) or (X < 0) or (Y >= image_size[1]) or (Y < 0):
    #        continue

    #    X = int(X)
    #    Y = int(Y)

    #    null_image[Y, X] = 255

    #kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    #dilate = cv2.dilate(null_image, kernel, iterations=10)
    #ni = cv2.resize(dilate, (scan_size[0], scan_size[1]))

    
    cv2.imwrite("/home/xytron/data/image/background/"+str(cnt)+".jpg", background)
    cv2.imwrite("/home/xytron/data/image/road/"+str(cnt)+".jpg", road)
    wr.writerow([str(cnt), current_p[0], current_p[1], current_q[0], current_q[1], current_q[2], current_q[3]])
    #cv2.imwrite("/home/xytron/data/image/scan/"+str(i)+".jpg", ni)
    
    cv2.waitKey(1)
    rate.sleep()

f.close()

