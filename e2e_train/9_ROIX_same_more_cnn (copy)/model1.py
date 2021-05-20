import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

use_cuda = torch.cuda.is_available()

class end2end(nn.Module):
    
    def __init__(self):
        super(end2end, self).__init__()
        #200,66
        conv1 = nn.Conv2d(in_channels=3, out_channels=24, kernel_size=(5, 5), stride=(2, 2), padding=(0, 0), dilation=(1, 1))
        #98,31
        conv2 = nn.Conv2d(in_channels=24, out_channels=36, kernel_size=(5, 5), stride=(2, 2), padding=(0, 0), dilation=(1, 1)) 
        #47,14
        conv3 = nn.Conv2d(in_channels=36, out_channels=48, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0), dilation=(1, 1)) 
        #45,12
        conv4 = nn.Conv2d(in_channels=48, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0), dilation=(1, 1)) 
        #43,10
        conv5 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0), dilation=(1, 1)) 
        #41,8
        conv6 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0), dilation=(1, 1)) 
        #39,6
        conv7 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0), dilation=(1, 1)) 
        #37,4
        conv8 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0), dilation=(1, 1)) 
        #35,2




        self.conv_module = nn.Sequential(
            conv1, nn.ReLU(),
            conv2, nn.ReLU(),
            conv3, nn.ReLU(),
            conv4, nn.ReLU(),
            conv5, nn.ReLU(),
            conv6, nn.ReLU(),
            conv7, nn.ReLU(),
            conv8, nn.ReLU(),


        )
        
        fc1_1 = nn.Linear(17920, 5000)
        fc1_2 = nn.Linear(5000, 1000)
        fc1_3 = nn.Linear(1000, 200)
        fc1_4 = nn.Linear(200, 50)
        fc1_5 = nn.Linear(50,1)

        fc2_1 = nn.Linear(17920, 5000)
        fc2_2 = nn.Linear(5000, 1000)
        fc2_3 = nn.Linear(1000, 200)
        fc2_4 = nn.Linear(200, 50)
        fc2_5 = nn.Linear(50,1)

        fc3_1 = nn.Linear(17920, 5000)
        fc3_2 = nn.Linear(5000, 1000)
        fc3_3 = nn.Linear(1000, 200)
        fc3_4 = nn.Linear(200, 50)
        fc3_5 = nn.Linear(50,1)


        self.fc_module1 = nn.Sequential(
            fc1_1, nn.ReLU(),
            fc1_2, nn.ReLU(),
            fc1_3, nn.ReLU(),
            fc1_4, nn.ReLU(),
            fc1_5
        )

        self.fc_module2 = nn.Sequential(
            fc2_1, nn.ReLU(),
            fc2_2, nn.ReLU(),
            fc2_3, nn.ReLU(),
            fc2_4, nn.ReLU(),
            fc2_5
        )

        self.fc_module3 = nn.Sequential(
            fc3_1, nn.ReLU(),
            fc3_2, nn.ReLU(),
            fc3_3, nn.ReLU(),
            fc3_4, nn.ReLU(),
            fc3_5
        )

        
    def forward(self, x):
        common = self.conv_module(x)
        common = torch.flatten(common, start_dim=1)
        x = self.fc_module1(common)
        y = self.fc_module2(common)
        yaw = self.fc_module3(common)

        return x, y, yaw
    
