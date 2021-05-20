#!/usr/bin/env python

import torch
import torch.nn as nn
import torch.optim as optim
from model_separate import end2end

import rospy, cv2, csv, time, io
import numpy as np
from tf.transformations import euler_from_quaternion
from cv_bridge import CvBridge

from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image
from PIL import Image as Img
from PIL import ImageTk
import matplotlib.pyplot as plt 
import matplotlib.cm as cm

name = "test_model1"
pub_topic = "test_model"

cv_image = np.empty(shape=[0])
background = np.empty(shape=[0])
road = np.empty(shape=[0])

bridge = CvBridge()
current_p = [0, 0]
current_q = [0,0,0,0]
rho = 0
phi = 0


model_path = "./town01_1.pth"
with open(model_path, 'rb') as f:
    LoadBuffer = io.BytesIO(f.read())
    
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

model = end2end().to(device)

model.load_state_dict(torch.load(LoadBuffer, map_location=device))
    
net = model
net.eval()

def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return rho, phi

def motor_callback(data):
    global current_p
    global current_q
    global rho, phi
    current_p = [data.pose.pose.position.x, data.pose.pose.position.y]
    current_q = [data.pose.pose.orientation.x, data.pose.pose.orientation.y, data.pose.pose.orientation.z, data.pose.pose.orientation.w]
    rho, phi = cart2pol(current_p[0], current_p[1])

def camera_callback(data):
    global cv_image, bridge
    global road, background
    cv_image = bridge.imgmsg_to_cv2(data, "bgr8")
    
    
def e2e_result(net, road, background, current_q):
    #get model
    road = np.array(road, 'uint8')
    bg = np.array(background, 'uint8')
    
    road = [[road.tolist()]]
    bg = [[bg.tolist()]]
    
    road = torch.FloatTensor(road).to(device)
    bg = torch.FloatTensor(bg).to(device)
    qt = torch.FloatTensor(current_q).to(device)
    
    s = road.shape
    ss = bg.shape
    #print("S", s)
    #print("S", ss)

    if (s[1] != 1) or (s[2] != 112) or (s[3] != 200):
        print("get out road")
        return float('inf'), float('inf')
    if (ss[1] != 1) or (ss[2] != 112) or (ss[3] != 200):
        print("get out bg")
        return float('inf'), float('inf')
    qt = torch.unsqueeze(qt,0)
    #print "road : ", road.shape, "bg : ", bg.shape, "qt : ", qt.shape, 
    output_rho, output_phi = net(road,bg,qt)
    output_rho = output_rho.item()
    output_phi = output_phi.item()

    return output_rho, output_phi
 
    
rospy.init_node(name, anonymous=True)
rospy.Subscriber("/carla/ego_vehicle/rgb_front/image", Image, camera_callback)
rospy.Subscriber("/carla/ego_vehicle/odometry", Odometry, motor_callback)

rate = rospy.Rate(10)
while True:
    if current_p == [0, 0]:
        continue
    if cv_image.size != (800 * 600 * 3):
        continue
    print("start")
    break

cnt =0
real =[]
e2e = []
colors = cm.rainbow(np.linspace(0, 1, 500))

while not rospy.is_shutdown():
    cv2.imshow("hehe", cv_image)
    
    cv_grayscale = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

    background = cv_grayscale[:240,:]
    background = cv2.resize(background, dsize=(200, 112))
        
    road = cv_grayscale[240:,:]
    road = cv2.resize(road, dsize=(200, 112))
    
    road = cv2.GaussianBlur(road,(5, 5), 0)
    road = cv2.Canny(np.uint8(road), 60, 70)

    #print("result" , e2e_result(net, road, background, current_q))
    erho, ephi = e2e_result(net, road, background, current_q)

    if erho == float('inf'):
        print("get out")
        continue

    print "---------------------------------------------------------------"
    print "real : rho - ", rho, "phi - ", phi
    print "e2e : rho - ", erho, "phi - ", ephi 
    print "err ; rho - ", rho - erho, "phi - ", phi - ephi
    
    plt.scatter(erho,ephi,c=colors[cnt], s = 2)
    plt.scatter(rho, phi, c = colors[cnt] , s = 2)
    
    
    cnt+=1
    if cnt == 300:
        break
    rate.sleep()
 
plt.show()

