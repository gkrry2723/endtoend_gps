#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.optim as optim

from model2 import end2end

import glob, csv, random, time, io, dill, os

import numpy as np
from PIL import Image

def study_model_save_x(epoch, batch_cnt, model):
    if not os.path.isdir("./save/"):
        os.mkdir("./save/")

    SavePath_main = os.getcwd()+"/save/main_model_"+str(epoch).zfill(6)+"_"+str(batch_cnt).zfill(6)+"net_x.pth"   
    #SavePath_main = os.getcwd()+"/save/main_model_"+str(epoch).zfill(6)+"_"+str(batch_cnt).zfill(6)+".pth"
    SaveBuffer = io.BytesIO()
    torch.save(model.state_dict(), SaveBuffer, pickle_module=dill)
    with open(SavePath_main, "wb") as f:
        f.write(SaveBuffer.getvalue())

def study_model_save_y(epoch, batch_cnt, model):
    if not os.path.isdir("./save/"):
        os.mkdir("./save/")

    SavePath_main = os.getcwd()+"/save/main_model_"+str(epoch).zfill(6)+"_"+str(batch_cnt).zfill(6)+"net_y.pth"   
    #SavePath_main = os.getcwd()+"/save/main_model_"+str(epoch).zfill(6)+"_"+str(batch_cnt).zfill(6)+".pth"
    SaveBuffer = io.BytesIO()
    torch.save(model.state_dict(), SaveBuffer, pickle_module=dill)
    with open(SavePath_main, "wb") as f:
        f.write(SaveBuffer.getvalue())

def study_model_save_yaw(epoch, batch_cnt, model):
    if not os.path.isdir("./save/"):
        os.mkdir("./save/")

    SavePath_main = os.getcwd()+"/save/main_model_"+str(epoch).zfill(6)+"_"+str(batch_cnt).zfill(6)+"net_yaw.pth"   
    #SavePath_main = os.getcwd()+"/save/main_model_"+str(epoch).zfill(6)+"_"+str(batch_cnt).zfill(6)+".pth"
    SaveBuffer = io.BytesIO()
    torch.save(model.state_dict(), SaveBuffer, pickle_module=dill)
    with open(SavePath_main, "wb") as f:
        f.write(SaveBuffer.getvalue())

def study_model_load(episode, model, cnt, device):
    LoadPath_main = os.getcwd()+"/save/main_model_"+str(episode).zfill(6)+"_"+str(cnt).zfill(6)+".pth"
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

batch_size = 100

x_batch = []
y_x_batch = []
y_y_batch = []
y_yaw_batch = []


epochs = 1000
epoch = 1
cnt = 1

net_x = end2end().to(device)
net_y = end2end().to(device)
net_yaw = end2end().to(device)
#net = study_model_load(202, net, 82, device).to(device)

loss_fn_x = nn.MSELoss()
loss_fn_y = nn.MSELoss()
loss_fn_yaw = nn.MSELoss()
optimizer_x = torch.optim.Adam(net_x.parameters(), lr=1e-4)
optimizer_y = torch.optim.Adam(net_y.parameters(), lr=1e-4)
optimizer_yaw = torch.optim.Adam(net_yaw.parameters(), lr=1e-4)




while (epoch < epochs):
    bc = 0
    random.shuffle(csv_data)
    for ccss in csv_data:
        if (cnt % batch_size) == 0:
            cnt = 1
            x = torch.FloatTensor(x_batch).to(device)
            y_x = torch.FloatTensor(y_x_batch).to(device)
            y_y = torch.FloatTensor(y_y_batch).to(device)
            y_yaw = torch.FloatTensor(y_yaw_batch).to(device)

            optimizer_x.zero_grad()
            optimizer_y.zero_grad()
            optimizer_yaw.zero_grad()
            output_x = net_x(x)
            output_y = net_y(x)
            output_yaw = net_yaw(x)
            loss_x = loss_fn_x(output_x, y_x)
            loss_y = loss_fn_y(output_y, y_y)
            loss_yaw = loss_fn_yaw(output_yaw, y_yaw)
            loss_x.backward()
            loss_y.backward()
            loss_yaw.backward()

            optimizer_x.step()
            optimizer_y.step()
            optimizer_yaw.step()

            x_batch = []
            y_x_batch = []
            y_y_batch = []
            y_yaw_batch = []
            bc += 1
        name = "image/"+ccss[1]+"-"+ccss[0]+".jpg"
        img = Image.open(name)
        img = img.convert('YCbCr')
        img = np.array(img)
        img = img.transpose((2, 0, 1)) / 255.0
        img = x_batch.append(img.tolist())
        label_x = y_x_batch.append([float(ccss[2])])
        label_y = y_y_batch.append([float(ccss[3])])
        label_yaw = y_yaw_batch.append([float(ccss[4])])

        cnt += 1
        #time.sleep(0.02)

    print("epoch : {} / {} | loss_x : {} | loss_y : {} | loss_yaw : {}".format(epoch, epochs, loss_x / 100, loss_y / 100, loss_yaw / 100))
    time.sleep(1)
    epoch += 1

    if (epoch % 100) == 0:
        study_model_save_x(epoch, cnt, net_x)
        study_model_save_y(epoch, cnt, net_y)
        study_model_save_yaw(epoch, cnt, net_yaw)
