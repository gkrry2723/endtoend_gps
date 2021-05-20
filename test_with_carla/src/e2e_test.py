#!/usr/bin/env python
# for model 1

import torch
import torch.nn as nn
import torch.optim as optim
from model_circular import end2end

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
bridge = CvBridge()
current_p = [0, 0]
current_q = [0,0,0,0]
roll, pitch, yaw = 0,0,0

#model_path = "./ROI_Same_hj.pth"
model_path = "./circular.pth"
with open(model_path, 'rb') as f:
    LoadBuffer = io.BytesIO(f.read())
    
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

model = end2end().to(device)

model.load_state_dict(torch.load(LoadBuffer, map_location=device))
    
net = model
net.eval()

def motor_callback(data):
    global current_p
    global current_q
    global roll, pitch, yaw
    current_p = [data.pose.pose.position.x, data.pose.pose.position.y]
    current_q = [data.pose.pose.orientation.x, data.pose.pose.orientation.y, data.pose.pose.orientation.z, data.pose.pose.orientation.w]
    (roll, pitch, yaw) = euler_from_quaternion(current_q)

def camera_callback(data):
    global cv_image
    global bridge
    cv_image = bridge.imgmsg_to_cv2(data, "bgr8")
    cv_image = cv2.resize(cv_image, dsize=(200, 112))
    cv_image = cv_image[46:,:]
    
def e2e_result(net, img):
    #get model
    img = Img.fromarray(img)
    img = img.convert('YCbCr')
    img = np.array(img)
    img = img.transpose((2, 0, 1)) / 255.0
    img = [img.tolist()]
    
    x = torch.FloatTensor(img).to(device)
    #print(size(x))
    s = x.shape
    if (s[0] != 1) or (s[1] != 3) or (s[2] != 66) or (s[3] != 200):
        print("get out")
        return float('inf'), float('inf'), float('inf')

    output_x, output_y, output_yaw = net(x)
    output_x = output_x.item()
    output_y = output_y.item()
    output_yaw = output_yaw.item()
    return output_x, output_y, output_yaw
    import matplotlib.cm as cm
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
colors = cm.rainbow(np.linspace(0, 1, 1200))

while not rospy.is_shutdown():
    cv2.imshow("hehe", cv_image)
    ex, ey, eyaw = e2e_result(net, cv_image)

    if ex == float('inf'):
        print("get out", ex, ey, eyaw)
        continue

    print "---------------------------------------------------------------"
    print "real : x - ", current_p[0], "y - ", current_p[1], "yaw - ", yaw
    print "e2e : x - ", ex, "y - ", ey, "yaw - ", eyaw 
    print "err ; x - ", current_p[0] - ex, "y - ", current_p[1] - ey, "yaw - ", yaw - eyaw
    
    
    plt.scatter(ex,ey,c=colors[cnt], s = 2)
    plt.scatter(current_p[0], current_p[1], c = colors[cnt] , s = 2)
    
    
    cnt+=1
    if cnt == 1000:
        break
    rate.sleep()
 
plt.show()

