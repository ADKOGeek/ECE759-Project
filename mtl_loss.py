import torch.nn as nn
import torch

#Custom loss for multitask learning. Is a weighted sum of regression and classification tasks.
#Note: Loss function has yet to be tested. It will probably yell and scream until we try actually using it
class MTL_Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = torch.asarray([0.2, 0.2, 0.2, 0.2, 0.2]) #weights set to be equal for now
        self.cross_loss = nn.CrossEntropyLoss()


    def forward(self, prediction, labels):
        regression_diffs = torch.sum(torch.abs(labels[:,1:] - prediction[:,1:]), dim=0) #L1 loss between predicted radar parameters
        classification_diffs = torch.unsqueeze(self.cross_loss(labels[:,0], prediction[:,0]), dim=0) #CE loss between classification picks
        diffs = torch.concat((regression_diffs, classification_diffs), dim=0) #append classification CE loss to regression L1 loss
        weighted = torch.multiply(diffs, self.weights) #apply weights
        total_loss = torch.sum(weighted) #sum losses

        return total_loss
