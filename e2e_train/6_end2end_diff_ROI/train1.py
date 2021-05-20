#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.optim as optim

from model1 import end2end

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

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

batch_size = 100

x_batch = []
y_x_batch = []
y_y_batch = []
y_yaw_batch = []


epochs = 1000
epoch = 1
cnt = 1

net= end2end().to(device)
#net = study_model_load(202, net, 82, device).to(device)

loss_fn_x = nn.MSELoss()
loss_fn_y = nn.MSELoss()
loss_fn_yaw = nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)



while (epoch < epochs):
    bc = 0
    random.shuffle(csv_data)
    for ccss in csv_data:
        if (cnt % batch_size) == 0:
            cnt = 1
            print(x_batch)
            exit()
            x = torch.FloatTensor(x_batch).to(device)
            y_x = torch.FloatTensor(y_x_batch).to(device)
            y_y = torch.FloatTensor(y_y_batch).to(device)
            y_yaw = torch.FloatTensor(y_yaw_batch).to(device)

            optimizer.zero_grad()
            output_x, output_y, output_yaw = net(x)
            loss_x = loss_fn_x(output_x, y_x)
            loss_y = loss_fn_y(output_y, y_y)
            loss_yaw = loss_fn_yaw(output_yaw, y_yaw)
            loss = loss_x + loss_y + loss_yaw
            loss.backward()
            optimizer.step()
            
            #print("epoch : {} / {} | step : {} / 24 | loss : {}".format(epoch, epochs, bc, loss / 100))

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

    print("epoch : {} / {} | loss : {}".format(epoch, epochs, loss / 100))
    time.sleep(1)
    epoch += 1

    if (epoch % 100) == 0:
        study_model_save(epoch, cnt, net)
