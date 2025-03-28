from data.load_radchar import load_data
from CNN_model import CNN_Model
from mtl_loss import MTL_Loss
from train_test import train, test
from plot_results import plot_all
import torch

#set device for torch to use
device = 'cpu'

#set hyperparameters
batch_size = 16
learning_rate = 0.001
num_epochs = 10

# load data
train_loader, val_loader, test_loader = load_data(batch_size=batch_size)

#define loss function and optimizer
loss_func = MTL_Loss()
optimizer = torch.optim.Adam(lr=learning_rate)

#load model
model = CNN_Model()

#train model
results = train(num_epochs, model, loss_func, optimizer, train_loader, val_loader, device)

#get model performance on test set
test_losses, test_acc = test(model, loss_func, test_loader)

#plot results
results = results.append(test_losses).append(test_acc)
plot_all(results)

