import torch
import torch.nn as nn
import torch.nn.functional as F

# Originally (batch_size, 1024)
# x = x.reshape(batch_size, 1, 32, 32)

# outputs radar_params (4 values) and p_type (5 softmaxed class probabilities)
# 4 values: num_pulses, pulse_width, time_delay, repetition_interval    
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
            nn.Dropout(0.25)
            # nn.Flatten()
        )

        # self.flatten = nn.Flatten()

        # 8 channels * 256 length = 2048
        self.fc = nn.Sequential(
            nn.Flatten(),               
            nn.Linear(2048, 128),     
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        # Task-specific heads
        self.num_pulses = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(32, 1)
        )

        self.pulse_width = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(32, 1)
        )

        self.time_delay = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(32, 1)
        )

        self.rep_interval = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(32, 1)
        )

        self.classifier = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(32, 5),
            nn.Softmax(dim=1)
        )


    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, 1, 32, 32)
        x = self.backbone(x)
        # x = self.flatten(x)
        x = self.fc(x)
        
        pulses = self.num_pulses(x)
        pulse_width = self.pulse_width(x)
        time = self.time_delay(x)
        repetition_interval = self.rep_interval(x)
        p_type = self.classifier(x)
        
        radar_params = torch.cat((pulses, pulse_width, time, repetition_interval), dim=1)
        return p_type, radar_params
