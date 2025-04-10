from matplotlib import pyplot as plt
from data.load_radchar import load_data


train_loader, val_loader, test_loader = load_data(batch_size=1)

dataiter = iter(train_loader)
sample = next(dataiter)
signal = sample['data'].squeeze(0).detach().cpu()
rad_params = sample['rad_params']
class_label = sample['class_label'].squeeze().detach().cpu()
print(f"Class label: {class_label}")

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

fig, ax = plt.subplots(1,2)
ax[0].plot(signal[0,:])
ax[0].set_title("I Component")
ax[1].plot(signal[1,:])
ax[1].set_title("Q Component")
fig.suptitle(f"Plot of class: {p_type}")
plt.show()