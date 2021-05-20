import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

use_cuda = torch.cuda.is_available()

class end2end(nn.Module):
    
    def __init__(self):
        super(end2end, self).__init__()
        #200,112 
        conv1 = nn.Conv2d(in_channels=1, out_channels=24, kernel_size=(5, 5), stride=(2, 2), padding=(0, 0), dilation=(1, 1))
        #98, 54
        conv2 = nn.Conv2d(in_channels=24, out_channels=36, kernel_size=(5, 5), stride=(2, 2), padding=(0, 0), dilation=(1, 1)) 
        #47, 25
        conv3 = nn.Conv2d(in_channels=36, out_channels=48, kernel_size=(5, 5), stride=(2, 2), padding=(0, 0), dilation=(1, 1)) 
        #22, 11
        conv4 = nn.Conv2d(in_channels=48, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0), dilation=(1, 1)) 
        #20, 9
        conv5 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0), dilation=(1, 1))
        #18, 7
        conv6 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0), dilation=(1, 1))
        #16, 5
        conv7 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0), dilation=(1, 1)) 
        #14, 3
        conv8 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0), dilation=(1, 1))
        #12, 1
        #conv9 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0), dilation=(1, 1))
        #13,2
        #3072

        conv21 = nn.Conv2d(in_channels=1, out_channels=24, kernel_size=(5, 5), stride=(2, 2), padding=(0, 0), dilation=(1, 1))
        conv22 = nn.Conv2d(in_channels=24, out_channels=36, kernel_size=(5, 5), stride=(2, 2), padding=(0, 0), dilation=(1, 1)) 
        conv23 = nn.Conv2d(in_channels=36, out_channels=48, kernel_size=(5, 5), stride=(2, 2), padding=(0, 0), dilation=(1, 1)) 
        conv24 = nn.Conv2d(in_channels=48, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0), dilation=(1, 1)) 
        conv25 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0), dilation=(1, 1))
        conv26 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0), dilation=(1, 1))
        conv27 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0), dilation=(1, 1)) 
        conv28 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0), dilation=(1, 1))
        #conv29 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0), dilation=(1, 1))
        
        self.conv_road = nn.Sequential(
            conv1, nn.ReLU(),
            conv2, nn.ReLU(),
            conv3, nn.ReLU(),
            conv4, nn.ReLU(),
            conv5, conv6, nn.ReLU(),
            conv7, conv8, nn.ReLU(),
            #conv9, nn.ReLU()
        )
        
        self.conv_background = nn.Sequential(
            conv21, nn.ReLU(),
            conv22, nn.ReLU(),
            conv23, nn.ReLU(),
            conv24, nn.ReLU(),
            conv25, conv26, nn.ReLU(),
            conv27, conv28, nn.ReLU(),
            #conv29, nn.ReLU()
        )

        fc1_1 = nn.Linear(6148, 3000)
        fc1_2 = nn.Linear(3000, 1000)
        fc1_3 = nn.Linear(1000, 100)
        fc1_4 = nn.Linear(100, 50)
        fc1_5 = nn.Linear(50, 10)
        fc1_6 = nn.Linear(10, 1) 

        fc2_1 = nn.Linear(6148, 3000)
        fc2_2 = nn.Linear(3000, 1000)
        fc2_3 = nn.Linear(1000, 100)
        fc2_4 = nn.Linear(100, 50)
        fc2_5 = nn.Linear(50, 10)
        fc2_6 = nn.Linear(10, 1) 


        self.fc_module1 = nn.Sequential(
            fc1_1, nn.ReLU(),
            fc1_2, nn.ReLU(),
            fc1_3, nn.ReLU(),
            fc1_4, nn.ReLU(),
            fc1_5, nn.ReLU(),
            fc1_6
        )

        self.fc_module2 = nn.Sequential(
            fc2_1, nn.ReLU(),
            fc2_2, nn.ReLU(),
            fc2_3, nn.ReLU(),
            fc2_4, nn.ReLU(),
            fc2_5, nn.ReLU(),
            fc2_6
        )

        
    def forward(self, road_img, bg_img, quat):

        road = self.conv_road(road_img)
        background = self.conv_background(bg_img)

        road = torch.flatten(road, start_dim=1)
        background = torch.flatten(background, start_dim=1)

        common = torch.cat([road,background,quat], dim=1)

        r = self.fc_module1(common)
        theta = self.fc_module2(common)

        return r, theta
    
