#########################################################
# Custom multi-task learning loss. Is weighted average of regression and classification tasks' losses.
#########################################################

import torch.nn as nn
import torch

#Custom loss for multitask learning. Is a weighted sum of regression and classification tasks.
#Note: Loss function has yet to be tested. It will probably yell and scream until we try actually using it
class MTL_Loss(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.weights = torch.asarray([1, 0, 0, 0, 0]).to(device) #weights for task (Class, then regression tasks) default: 0.2 each
        self.cross_loss = nn.CrossEntropyLoss()


    def forward(self, class_pred, rad_pred, class_actual, rad_actual):
        regression_diffs = torch.sum(torch.abs(rad_pred - rad_actual), dim=0) #L1 loss between predicted radar parameters
        classification_diffs = torch.unsqueeze(self.cross_loss(class_pred, class_actual), dim=0) #CE loss between classification picks
        diffs = torch.concat((classification_diffs,regression_diffs), dim=0) #append classification CE loss to regression L1 loss
        weighted = torch.multiply(diffs, self.weights) #apply weights
        total_loss = torch.sum(weighted) #sum losses

        return total_loss
