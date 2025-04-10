#########################################################
# Placeholder MLP network for testing data loading, training, testing, plotting, etc.
#########################################################

import torch
import torch.nn as nn
#task head for each estimated parameter
class TaskHead(nn.Module):
    def __init__(self, embed_dim, dropout=0):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=1,out_channels=embed_dim, kernel_size=3, padding=1, padding_mode='circular'),
            #nn.GELU(),
            #nn.Dropout(p=dropout)
            )
        self.lin_out = nn.Linear(embed_dim*embed_dim,1)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv(x)
        x = x.reshape(x.shape[0], -1)
        x = self.lin_out(x)

        return x

class Placeholder_MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(1024,512),
            nn.GELU(),
            nn.Linear(512,256),
            nn.GELU(),
            nn.Linear(256,128),
            nn.GELU(),
            nn.Linear(128,64),
            nn.GELU()
        )
        #self.output = nn.Linear(64,5)


        self.num_pulses = TaskHead(64)
        self.pulse_width = TaskHead(64)
        self.time_delay = TaskHead(64)
        self.repetition_interval = TaskHead(64)
        self.classifier = nn.Sequential(
            nn.Linear(64,5), #5 classes
            nn.Softmax(dim=1)
        )


    def forward(self, x):
        batch_size = x.shape[0]
        x = x.reshape(batch_size, 1024)
        x = self.shared(x)
        np = self.num_pulses(x)
        pw = self.pulse_width(x)
        td = self.time_delay(x)
        ri = self.repetition_interval(x)
        p_type = self.classifier(x)
        rad_params = torch.cat((np,pw,td,ri), dim=1)
        return p_type, rad_params
