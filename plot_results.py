#########################################################
# Helper function for plotting all of the training and validation results
#########################################################

from matplotlib import pyplot as plt
import matplotlib
import numpy as np

def plot_all(results):
    plt.plot(results[0])
    plt.plot(results[2])
    plt.ylim(0,15)
    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MTL Loss")
    plt.legend(["Training", "Validation"])
    plt.show()

    plt.plot(results[1])
    plt.plot(results[3])
    plt.title("Training and Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend(["Training", "Validation"])
    plt.show()

if (__name__=="__main__"):
    font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 40}
    matplotlib.rc('font', **font)
    results = np.load('./results_CNN.npy')
    plot_all(results)
    
