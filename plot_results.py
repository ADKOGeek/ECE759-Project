#########################################################
# Helper function for plotting all of the training and validation results
#########################################################

from matplotlib import pyplot as plt
import numpy as np

def plot_all(results):
    plt.plot(results[0])
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MTL Loss")
    plt.show()

    plt.plot(results[1])
    plt.title("Training Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.show()

    plt.plot(results[2])
    plt.title("Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MTL Loss")
    plt.show()

    plt.plot(results[3])
    plt.title("Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.show()

if (__name__=="__main__"):
    results = np.load('./results_onlyClass_lightIQST.npy')
    plot_all(results)
    
