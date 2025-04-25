Important Note:

This repository has two branches. The 'main' branch was used for training the IQST while the 'CNN' branch was used to train the CNN.



Python files:
Runnable stuff:
main.py is the main entry point for training the models. Running this file will load the Radchar dataset, train a neural net, save its parameter weights, and save the training and validation loss and accuracy curves under ./results.

MAE_on_set.py is a script used to plot the mean absolute error of the parameter regression tasks for a pre-trained model over a variety of signal to noise ratios on the test set. This script loads a model from a checkpoint and then checks its performance.

visualize_data.py is a script used to take a waveform from the RadChar dataset, plot the IQ components, and get the model prediction on signal type and signal parameter values.

plot_results.py is used to display the training and validation loss and accuracy curves. The main.py file also calls the function defined to plot the results after training.


Helper files:
data/load_radchar.py uses a custom PyTorch dataset class to load the baseband signals into a format that we can put into our neural network models. The file assumes that the RadChar dataset has been downloaded and placed into the ./data folder. Otherwise, the data will not load.

Model classes for the CNN and IQST can be found in the ./models folder in IQST.py and CNN.py respectively. The placeholder_MLP.py file is just a placeholder model that was used to develop code, so please ignore it.

./training_testing/train_test.py contains the main training and validation loops.

./training_testing/mtl_loss.py defines the custom multi-task loss function

