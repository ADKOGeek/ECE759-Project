############################################
# main.py: entry point for training neural network. Run this file to make it work.
#############################################

from data.load_radchar import load_data
from placeholder_MLP import Placeholder_MLP
from IQST import IQST
from mtl_loss import MTL_Loss
from train_test import train, test
from plot_results import plot_all
import torch


#set device for torch to use (doesn't actually do anything right now lol)
device = (
    torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda:0")
)
print("Using device", device)

#set hyperparameters
batch_size = 64
learning_rate = 0.0005
num_epochs = 100

# load data
train_loader, val_loader, test_loader = load_data(batch_size=batch_size)

#load model and init weights with LeCun initialization
model = IQST(device).to(device)
#model.apply(init_weights)

#define loss function and optimizer
loss_func = MTL_Loss(device)
optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)

#train model
results = train(num_epochs, model, loss_func, optimizer, train_loader, val_loader, device)

#get model performance on test set
test_losses, test_acc = test(model, loss_func, test_loader, device)

#display results
plot_all(results)
print(f"test loss: {test_losses}, test acc:{test_acc}")

