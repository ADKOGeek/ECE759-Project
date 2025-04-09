#########################################################
# File to implement training, validation, and test functions
#########################################################

import torch
from tqdm import tqdm

#training function, calculates all metrics during training
def train(num_epochs, model, loss_func, optimizer, train_loader, val_loader, device):
    train_losses = []
    val_losses = []
    train_accuracy = []
    val_accruacy = []

    for epoch in tqdm(range(num_epochs), desc="Training...", leave=False):
        train_loss, train_acc = train_one_epoch(model, loss_func, optimizer, train_loader, device)
        val_loss, val_acc = test(model, loss_func, val_loader, device)
        print(f"\nCompleted epoch {epoch}. train_acc:{train_acc}, val_acc:{val_acc}, train_loss:{train_loss}, val_loss:{val_loss}")

        #append metrics to arrays keeping track of them
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracy.append(train_acc)
        val_accruacy.append(val_acc)

    print("Training complete")
    return [train_losses, train_accuracy, val_losses, val_accruacy]

#performs one epoch of prediction + backpropagation
def train_one_epoch(model, loss_func, optimizer, dataloader, device):
    model.train()
    train_loss = 0
    correct = 0
    total_samples = 0

    for i, sample in tqdm(enumerate(dataloader), leave=False):
        signal = sample['data'].to(device)
        rad_params = sample['rad_params'].to(device)
        class_label = sample['class_label'].to(device)
        class_vec = torch.zeros(class_label.shape[0],5).to(device) #class labels are ints, so we need to rearrange into a one-hot vector for CE loss computation
        class_vec[torch.arange(class_vec.shape[0]),class_label] = 1

        #make model prediction
        class_pred, rad_pred = model(signal)

        #calculate loss
        loss = loss_func(class_pred, rad_pred, class_vec, rad_params)
        train_loss += loss.item()

        #perform optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #calculate number of correct predictions for epoch
        predicted = torch.argmax(class_pred, dim = 1)
        correct += (predicted == class_label).sum().item()
        total_samples += len(class_label)

    train_loss = train_loss / len(dataloader)
    accuracy = correct / total_samples * 100

    return train_loss, accuracy


#returns test loss and accuracy (also used for validation)
def test(model, loss_func, dataloader, device):
    model.eval()
    test_loss = 0
    correct = 0
    total_samples = 0

    with torch.no_grad():
        for i, sample in tqdm(enumerate(dataloader), leave=False):
            #extract data from dataloader
            signal = sample['data'].to(device)
            rad_params = sample['rad_params'].to(device)
            class_label = sample['class_label'].to(device)
            class_vec = torch.zeros(class_label.shape[0],5).to(device) #class labels are ints, so we need to rearrange into a one-hot vector for CE loss computation
            class_vec[torch.arange(class_vec.shape[0]),class_label] = 1
            
            #make prediction on model
            class_pred, rad_pred = model(signal)

            #calculate loss
            test_loss += loss_func(class_pred, rad_pred, class_vec, rad_params).item()

            #calculate number of correct predictions for epoch
            predicted = torch.argmax(class_pred, dim = 1)
            correct += (predicted == class_label).sum().item()
            total_samples += len(class_label)
    
    test_loss = test_loss / len(dataloader)
    accuracy = correct / total_samples * 100

    return test_loss, accuracy