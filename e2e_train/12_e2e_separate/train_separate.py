#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.optim as optim

from model_separate import end2end

import glob, csv, random, time, io, dill, os

import numpy as np
from PIL import Image

def study_model_save(epoch, batch_cnt, model):
    if not os.path.isdir("./save/"):
        os.mkdir("./save/")
    SavePath_main = os.getcwd()+"/save/main_model_"+str(epoch).zfill(6)+"_"+str(batch_cnt).zfill(6)+".pth"
    SaveBuffer = io.BytesIO()
    torch.save(model.state_dict(), SaveBuffer, pickle_module=dill)
    with open(SavePath_main, "wb") as f:
        f.write(SaveBuffer.getvalue())

def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return rho, phi

csv_files = glob.glob("*.csv")
csv_data = []
for csv_file in csv_files:
    f = open(csv_file, 'r')
    reader = csv.reader(f)
    reader.next()
    for row in reader:
        #                 num      x       y       qx      qy      qz      qw 
        csv_data.append((row[0], row[1], row[2], row[3], row[4], row[5], row[6]))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")

batch_size = 100

x_road_batch = []
x_bg_batch = []
x_qt_batch = []
y_r_batch = []
y_theta_batch = []

epochs = 4000
epoch = 1
cnt = 1

net= end2end().to(device)

loss_fn_r = nn.MSELoss()
loss_fn_theta = nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)

while (epoch < epochs):
    bc = 0
    random.shuffle(csv_data)
    for ccss in csv_data:
        if (cnt % batch_size) == 0:
            cnt = 1
            
            x_road = torch.FloatTensor(x_road_batch).to(device)
            x_bg = torch.FloatTensor(x_bg_batch).to(device)

            x_qt = torch.FloatTensor(x_qt_batch).to(device)
            
            y_r = torch.FloatTensor(y_r_batch).to(device)
            y_theta = torch.FloatTensor(y_theta_batch).to(device)

            optimizer.zero_grad()
            
            output_r, output_theta = net(x_road, x_bg, x_qt)
            loss_r = loss_fn_r(output_r, y_r)
            loss_theta = loss_fn_theta(output_theta, y_theta)
            loss = loss_r * 100 + loss_theta
            loss.backward()
            optimizer.step()

            x_road_batch = []
            x_bg_batch = []
            x_qt_batch = []
            y_r_batch = []
            y_theta_batch = []

            bc += 1
        road_name = "road/"+ccss[0]+".jpg"
        bg_name = "background/"+ccss[0]+".jpg"

        road_img = Image.open(road_name)
        bg_img = Image.open(bg_name)

        road_img = np.array(road_img, 'uint8')
        bg_img = np.array(bg_img, 'uint8')
        
        road_img = x_road_batch.append([road_img.tolist()])
        bg_img = x_bg_batch.append([bg_img.tolist()])
        x_qt_batch.append([float(ccss[3]), float(ccss[4]), float(ccss[5]), float(ccss[6])])
        
        pol_r, pol_theta = cart2pol(float(ccss[1]),float(ccss[2]))
        label_r = y_r_batch.append([pol_r])
        label_theta = y_theta_batch.append([pol_theta])        
        
        cnt += 1

    print("epoch : {} / {} | loss : {}".format(epoch, epochs, loss / 100))
    time.sleep(1)
    epoch += 1

    if (epoch % 100) == 0:
        study_model_save(epoch, cnt, net)
