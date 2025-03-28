import torch
import torch.nn as nn

class CNN_Model(nn.Module):
    def __init__(self, batch_size):
        super().__init__()
        self.batch_size = batch_size
        self.input = nn.Linear(1024, 512) #placeholder architecture with linear layers
        self.shared = nn.Sequential(
            nn.Linear(512,256),
            nn.Linear(256,128),
            nn.Linear(128,64),
        )
        self.output = nn.Linear(64,5)

        # self.num_pulses = nn.Sequential(
        #     nn.Conv1d(),
        #     nn.Linear()
        # )
        # self.pulse_width = nn.Sequential(
        #     nn.Conv1d(),
        #     nn.Linear()
        # )
        # self.time_delay = nn.Sequential(
        #     nn.Conv1d(),
        #     nn.Linear()
        # )
        # self.repetition_interval = nn.Sequential(
        #     nn.Conv1d(),
        #     nn.Linear()
        # )
        # self.classifier = nn.Sequential(
        #     nn.Conv1d(),
        #     nn.Linear(),
        #     nn.Softmax()
        # )


    def forward(self, x):
        x = x.reshape(self.batch_size, 1024)
        x = self.input(x)
        x = self.shared(x)
        x = self.output(x)
        return x
