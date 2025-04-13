import torch
import torch.nn as nn
import torch.nn.functional as F

# Originally (batch_size, 1024)
# x = x.reshape(batch_size, 1, 32, 32)

# outputs radar_params (4 values) and p_type (5 softmaxed class probabilities)
# 4 values: num_pulses, pulse_width, time_delay, repetition_interval    
class TaskHead(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=32, dropout=0.5):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x):
        return self.block(x)


class CNN_model(nn.Module):
    def __init__(self):
        super().__init__()
        # Paper uses a shared backbone
        self.backbone = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # nn.Conv2d(64, 128, kernel_size=3, padding=1),
            # nn.ReLU(),
            # nn.MaxPool2d(2),

            # nn.Conv2d(128, 256, kernel_size=3, padding=1),
            # nn.ReLU(),
            # nn.MaxPool2d(2),    

            nn.Dropout(0.25)
            # nn.Flatten()
        )

        # self.flatten = nn.Flatten()

        # 8 channels * 256 length = 2048
        self.fc = nn.Sequential(
            nn.Flatten(),       
            # nn.Linear(1024, 128),
            nn.Linear(1024, 128),     
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        # # Task-specific heads
        # self.num_pulses = nn.Sequential(
        #     nn.Linear(128, 32),
        #     nn.ReLU(),
        #     nn.Dropout(0.5),
        #     nn.Linear(32, 1)
        # )

        # self.pulse_width = nn.Sequential(
        #     nn.Linear(128, 32),
        #     nn.ReLU(),
        #     nn.Dropout(0.5),
        #     nn.Linear(32, 1)
        # )

        # self.time_delay = nn.Sequential(
        #     nn.Linear(128, 32),
        #     nn.ReLU(),
        #     nn.Dropout(0.5),
        #     nn.Linear(32, 1)
        # )

        # self.repetition_interval = nn.Sequential(
        #     nn.Linear(128, 32),
        #     nn.ReLU(),
        #     nn.Dropout(0.5),
        #     nn.Linear(32, 1)
        # )
        self.num_pulses = TaskHead()
        self.pulse_width = TaskHead()
        self.time_delay = TaskHead()
        self.rep_interval = TaskHead()


        self.classifier = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(32, 5),
            nn.Softmax(dim=1)
        )


    def forward(self, x):
        batch_size = x.size(0)
        x.view(batch_size, -1)
        x = x.view(batch_size, 1, 32, 32)
        x = self.backbone(x)
        # x = self.flatten(x)
        x = self.fc(x)
        
        np = self.num_pulses(x)
        pw = self.pulse_width(x)
        td = self.time_delay(x)
        ri = self.rep_interval(x)
        p_type = self.classifier(x)
        rad_params = torch.cat((np,pw,td,ri), dim=1)
        
        radar_params = torch.cat((np, pw, td, ri), dim=1)
        return p_type, radar_params
