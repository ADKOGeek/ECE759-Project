from matplotlib import pyplot as plt

def plot_all(results):
    plt.plot(results[0])
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MTL Loss")
    plt.show()

    plt.plot(results[1])
    plt.title("Training Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("MTL Loss")
    plt.show()

    plt.plot(results[2])
    plt.title("Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MTL Loss")
    plt.show()

    plt.plot(results[3])
    plt.title("Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("MTL Loss")
    plt.show()

    plt.plot(results[4])
    plt.title("Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MTL Loss")
    plt.show()

    plt.plot(results[5])
    plt.title("Test Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("MTL Loss")
    plt.show()
