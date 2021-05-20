#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.optim as optim

from model1 import end2end

import glob, csv, random, time, io, dill, os, cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt 


def study_model_load(episode, batch_cnt, model, device):
    LoadPath_main = os.getcwd()+"/save__/main_model_001000_000010.pth" #"+str(episode).zfill(6)+  str(batch_cnt).zfill(6)+ ".pth"
    with open(LoadPath_main, 'rb') as f:
        LoadBuffer = io.BytesIO(f.read())
    model.load_state_dict(torch.load(LoadBuffer, map_location=device))
    return model

csv_files = glob.glob("*.csv")
csv_data = []
for csv_file in csv_files:
    f = open(csv_file, 'r')
    reader = csv.reader(f)
    reader.next()
    for row in reader:
        csv_data.append((csv_file[:-4], row[1], row[2], row[3], row[4]))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

net = end2end().to(device)
net = study_model_load(1100, 73, net, device)
net.eval()

#x_batch = []
#y_x_batch = []
#y_y_batch = []
#y_yaw_batch = []
cnt = 1

x_sum = 0
y_sum = 0
yaw_sum = 0

for ccss in csv_data:
    name = "image/"+ccss[1]+"-"+ccss[0]+".jpg"
    img = Image.open(name)
    #img.show()
    
    
    img = img.convert('YCbCr')
    img = np.array(img)
    img = img.transpose((2, 0, 1)) / 255.0
    img = [img.tolist()]

    x = torch.FloatTensor(img).to(device)
    
    output_x, output_y, output_yaw = net(x)
    output_x = output_x.item()
    output_y = output_y.item()
    output_yaw = output_yaw.item() 
    
    print "---------------------------------------------------------------"
    print "real : x - ", ccss[2], "y - ", ccss[3], "yaw - ", ccss[4]
    print "e2e : x - ", output_x, "y - ", output_y, "yaw - ", output_yaw 

    #x_acc = abs(output_x - float(ccss[2])) / float(ccss[2]) *100
    #y_acc = abs(output_y - float(ccss[3])) / float(ccss[3]) *100
    #yaw_acc = abs(output_yaw - float(ccss[4])) / float(ccss[4]) *100

    #x_sum += x_acc
    #y_sum += y_acc
    #yaw_sum += yaw_acc
    
    #plt.scatter(output_x, output_y,c= "red", s = 2)
    plt.scatter(ccss[2], ccss[3], c = "blue", s = 2)
    
    cnt += 1

    #print("epoch : {} / {} | x_loss : {} | y_loss : {} | yaw_loss : {}".format(cnt, len(csv_data), x_acc, y_acc, yaw_acc))


print("Finish")
#print("avg acc : x - {}, y - {}, yaw - {}".format(x_sum/float(cnt), y_sum/float(cnt), yaw_sum/float(cnt)))
plt.show()