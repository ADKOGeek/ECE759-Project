from matplotlib import pyplot as plt
import matplotlib
from data.load_radchar import load_data
from models.IQST import IQST
import torch

def label_to_string(class_label):
    p_type = ""
    if (class_label == 0):
        p_type = "Coherent Pulse Train"
    elif (class_label == 1):
        p_type = "Barker Code"
    elif (class_label == 2):
        p_type = "Polyphase Barker Code"
    elif (class_label == 3):
        p_type = "Frank Code"
    elif (class_label == 4):
        p_type = "Linear Frequency Modulated"
    else:
        p_type = "This is not a class"

    return p_type





#set device for torch to use (doesn't actually do anything right now lol)
device = (
    torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda:0")
)
print("Using device", device)

font = {'weight' : 'bold',
        'size'   : 40}
matplotlib.rc('font', **font)

#load data sample
train_loader, val_loader, test_loader = load_data(batch_size=1)
dataiter = iter(test_loader)
sample = next(dataiter)
signal = sample['data']
rad_params = sample['rad_params'].squeeze().detach().cpu()
class_label = sample['class_label'].squeeze().detach().cpu()
p_type = label_to_string(class_label)

#make prediction on batch
model = IQST(device).to(device)
model.load_state_dict(torch.load('./checkpoints/IQST_SmallData.pth', map_location=torch.device('cpu'))) #load model from pth file
model.eval()
class_pred, param_pred = model(signal)

pred_label = torch.argmax(class_pred) #get class index prediction
pred_ptype = label_to_string(pred_label)
pred_params = param_pred.squeeze().detach().cpu() #get predicted radar params

#get signal off of gpu and into a tensor that can be graphed
signal = signal.squeeze(0).detach().cpu()

#plot signal
fig, ax = plt.subplots(1,2)
ax[0].plot(signal[0,:])
ax[0].set_title("I Component")
ax[0].set_xlabel("Time")
ax[0].set_ylabel("Amplitude")
ax[1].plot(signal[1,:])
ax[1].set_title("Q Component")
ax[1].set_xlabel("Time")
ax[1].set_ylabel("Amplitude")
fig.suptitle(f"Actual signal type: {p_type}, Predicted signal type: {pred_ptype}")
plt.show()

#print true vs predicted radar parameters
print(f"Normalized true params: np-{rad_params[0]}, pw-{rad_params[1]}, td-{rad_params[2]}, pri-{rad_params[3]}")
print(f"Normalized predicted params: np-{pred_params[0]}, pw-{pred_params[1]}, td-{pred_params[2]}, pri-{pred_params[3]}")

