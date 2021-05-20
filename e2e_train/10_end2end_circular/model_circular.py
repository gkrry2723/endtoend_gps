import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.tensor as tensor

use_cuda = torch.cuda.is_available()

class end2end(nn.Module):
    
    def __init__(self):
        super(end2end, self).__init__()
        conv1 = nn.Conv2d(in_channels=3, out_channels=24, kernel_size=(5, 5), stride=(2, 2), padding=(0, 0), dilation=(1, 1))
        conv2 = nn.Conv2d(in_channels=24, out_channels=36, kernel_size=(5, 5), stride=(2, 2), padding=(0, 0), dilation=(1, 1)) 
        conv3 = nn.Conv2d(in_channels=36, out_channels=48, kernel_size=(5, 5), stride=(2, 2), padding=(0, 0), dilation=(1, 1)) 
        conv4 = nn.Conv2d(in_channels=48, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0), dilation=(1, 1)) 
        conv5 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0), dilation=(1, 1)) 
        
        self.conv_module = nn.Sequential(
            conv1, nn.ReLU(),
            conv2, nn.ReLU(),
            conv3, nn.ReLU(),
            conv4, nn.ReLU(),
            conv5, nn.ReLU(),
        )
        
        #-------------------------------------------------------------------------------------------------------------------------------
        fc1_1 = nn.Linear(1152, 100)
        fc1_2 = nn.Linear(100, 50)
        fc1_3 = nn.Linear(50, 10)
        fc1_4 = nn.Linear(10, 1)

        fc2_1 = nn.Linear(1153, 100)
        fc2_2 = nn.Linear(100, 50)
        fc2_3 = nn.Linear(50, 10)
        fc2_4 = nn.Linear(10, 1)

        fc3_1 = nn.Linear(1154, 100)
        fc3_2 = nn.Linear(100, 50)
        fc3_3 = nn.Linear(50, 10)
        fc3_4 = nn.Linear(10, 1)


        self.fc_module_x = nn.Sequential(
            fc1_1, nn.ReLU(),
            fc1_2, nn.ReLU(),
            fc1_3, nn.ReLU(),
            fc1_4
        )

        self.fc_module2 = nn.Sequential(
            fc2_1, nn.ReLU(),
            fc2_2, nn.ReLU(),
            fc2_3, nn.ReLU(),
            fc2_4
        )

        self.fc_module3 = nn.Sequential(
            fc3_1, nn.ReLU(),
            fc3_2, nn.ReLU(),
            fc3_3, nn.ReLU(),
            fc3_4 
        )

        
    def forward(self, x):
        common = self.conv_module(x)
        common = torch.flatten(common, start_dim=1)

        x = self.fc_module_x(common)
        new_input1 = torch.cat((common,x),1)

        y = self.fc_module2(new_input1)
        new_input2 = torch.cat((new_input1,y),1)

        yaw = self.fc_module3(new_input2)

        return x, y, yaw
    
