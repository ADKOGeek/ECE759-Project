from data.load_radchar import load_data
from CNN_model import CNN_Model
from mtl_loss import MTL_Loss
from train_loop import train

#set hyperparameters
batch_size = 16
learning_rate = 0.001

# load data
dataloader = load_data(batch_size=batch_size)

#define loss function
loss = MTL_Loss()

#load model
model = CNN_Model()

#train model
train()

#test model


#plot results?