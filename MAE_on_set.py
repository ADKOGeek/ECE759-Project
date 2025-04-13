from matplotlib import pyplot as plt
import matplotlib
from data.load_radchar import load_data
from IQST import IQST
from CNN import CNN_model
import torch
from tqdm import tqdm
import numpy as np

#set device for torch to use (doesn't actually do anything right now lol)
device = (
    torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda:0")
)
print("Using device", device)

#set font
font = {'weight' : 'bold',
        'size'   : 40}
matplotlib.rc('font', **font)

#load data sample
train_loader, val_loader, test_loader = load_data(batch_size=1)

#load models
model0 = IQST(device).to(device)
model0.load_state_dict(torch.load('./IQST_SmallData.pth', map_location=torch.device('cpu'))) #load model from pth file
model0.eval()
model1 = CNN_model().to(device)
model1.load_state_dict(torch.load('./CNN_1.pth', map_location=torch.device('cpu')))
model1.eval()
models = [model0, model1]

#init metrics
absolute_error = torch.zeros((1,4,2))
absolute_error_snr = torch.zeros((4,41,2))
absolute_error_snr_counts = torch.zeros((1,41,2))
accuracy = torch.zeros(2)
accuracy_snr = torch.zeros((1,41,2))
accuracy_snr_counts = torch.zeros((1,41,2))

count = 0
for j in range(0,2):
    count = 0
    correct = 0
    total_samples = 0
    #compute metrics over test dataset
    with torch.no_grad():
        for i, sample in tqdm(enumerate(test_loader), leave=False):
            #extract data from dataloader
            signal = sample['data'].to(device)
            rad_params = sample['rad_params'].to(device)
            class_label = sample['class_label'].to(device)
            snr = sample['snr']
            class_vec = torch.zeros(class_label.shape[0],5).to(device) #class labels are ints, so we need to rearrange into a one-hot vector for CE loss computation
            class_vec[torch.arange(class_vec.shape[0]),class_label] = 1
            
            #make prediction on model
            class_pred, rad_pred = models[j](signal)

            #calculate L1 error
            current_error = torch.abs(rad_pred - rad_params)
            absolute_error[:,:,j] += current_error

            #count number of correct class predictions
            predicted = torch.argmax(class_pred, dim = 1)
            current_correct = (predicted == class_label).sum().item()
            correct += current_correct
            total_samples += len(class_label)

            #add error to different index based on snr
            snr_index = snr + 20
            absolute_error_snr[:,snr_index,j] += torch.transpose(current_error, dim0=0, dim1=1)
            absolute_error_snr_counts[:,snr_index,j] += len(snr)
            accuracy_snr[:,snr_index,j] += current_correct
            accuracy_snr_counts[:, snr_index,j] += 1
            count += len(class_label)

    #compute overall accuracy
    accuracy[j] = correct / total_samples * 100

MAE = (absolute_error / count).squeeze()
MAE_snr = (torch.div(absolute_error_snr, absolute_error_snr_counts)*64).detach().cpu() #element-wise division
accuracy_snr = (torch.div(accuracy_snr, accuracy_snr_counts) * 100).squeeze(0).detach().cpu()
x_axis = np.squeeze((np.indices((1,41))[1] - 20), 0)
#plot MAE vs SNR
#plot signal
plt.plot(x_axis, MAE_snr[0,:,0])
plt.plot(x_axis, MAE_snr[0,:,1])
plt.title("MAE for Number of Pulses")
plt.xlabel("SNR (dB)")
plt.ylabel("MAE")
plt.legend(["IQST", "CNN"])
plt.show()

plt.plot(x_axis, MAE_snr[1,:,0])
plt.plot(x_axis, MAE_snr[1,:,1])
plt.title("MAE for Pulse Width")
plt.xlabel("SNR (dB)")
plt.ylabel("MAE")
plt.legend(["IQST", "CNN"])
plt.show()

plt.plot(x_axis, MAE_snr[2,:,0])
plt.plot(x_axis, MAE_snr[2,:,1])
plt.title("MAE for Time Delay")
plt.xlabel("SNR (dB)")
plt.ylabel("MAE")
plt.legend(["IQST", "CNN"])
plt.show()

plt.plot(x_axis, MAE_snr[3,:,0])
plt.plot(x_axis, MAE_snr[3,:,1])
plt.title("MAE for Pulse Repetition Interval")
plt.xlabel("SNR (dB)")
plt.ylabel("MAE")
plt.legend(["IQST", "CNN"])
plt.show()

plt.plot(x_axis, accuracy_snr[:,0])
plt.plot(x_axis, accuracy_snr[:,1])
plt.title("Accuracy vs SNR")
plt.xlabel("SNR (dB)")
plt.ylabel("Accuracy (%)")
plt.legend(["IQST", "CNN"])
plt.show()


print(f"MAE on test set:\n# pulses:{MAE[0]}, pulse width:{MAE[1]}, time delay:{MAE[2]}, pulse repetition interval:{MAE[3]}")
print(f"Accuracy on test set:{accuracy}")
