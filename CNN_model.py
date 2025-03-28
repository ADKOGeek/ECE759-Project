import torch
import torch.nn as nn

class CNN_Model(nn.Module):
    def __init__(self, batch_size):
        super().__init__()
        self.batch_size = batch_size
        self.shared = nn.Sequential(
            nn.Linear(1024,512),
            nn.Linear(512,256),
            nn.Linear(256,128),
            nn.Linear(128,64)
        )
        #self.output = nn.Linear(64,5)


        self.num_pulses = nn.Sequential(
            nn.Linear(64,1)
        )
        self.pulse_width = nn.Sequential(
            nn.Linear(64,1)
        )
        self.time_delay = nn.Sequential(
            nn.Linear(64,1)
        )
        self.repetition_interval = nn.Sequential(
            nn.Linear(64,1)
        )
        self.classifier = nn.Sequential(
            nn.Linear(64,5), #5 classes
            nn.Softmax(dim=1)
        )


    def forward(self, x):
        x = x.reshape(self.batch_size, 1024)
        x = self.shared(x)
        np = self.num_pulses(x)
        pw = self.pulse_width(x)
        td = self.time_delay(x)
        ri = self.repetition_interval(x)
        p_type = self.classifier(x)
        rad_params = torch.cat((np,pw,td,ri), dim=1)
        return p_type, rad_params
