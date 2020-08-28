import warnings
warnings.filterwarnings("ignore")
# a) Use the data processing code from the 1st exercise session to import the data
from Data_preprocessing import X_train, y_train, X_test, y_test

# b) Sklearn method
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

scores_train = []
scores_test = []
# c) Using a loop to try different numbers of hidden layers and neurons
for j in range(1,11):
    # j is the number of layers j=1,2,...,10
    # We initiate these lists to save the scores of each architecture
    neurons_scores_train = []
    neurons_scores_test = []
    for i in range(1,7):
        # i*5 is the number of neurons per layer e.g 5,10,15,...,30
        # layers are lists of number of neurons in each hidden layer (input and output layers don't count) for example (5,5,5) means 3 hidden layers with 5 neurons each
        layers = j*[i*5]
        # Initialise the classifier
        clf = MLPClassifier(solver='lbfgs',hidden_layer_sizes=layers)
        # Fit to the training data
        clf.fit(X_train,y_train)
        # Save the scores related to training and to test data
        neurons_scores_train.append(clf.score(X_train, y_train))
        neurons_scores_test.append(clf.score(X_test, y_test))
    # Save the list of scores related to this number of layers
    scores_train.append(neurons_scores_train)
    scores_test.append(neurons_scores_test)

scores_train = np.array(scores_train)
scores_test = np.array(scores_test)
print("Training scores: ",scores_train)
print("Test scores: ", scores_test)
# d) This is a 3D visualisation of the scores. This is of course just some extra demonstration, it is not mandatory in the exercise.
length = scores_train.shape[0]
width = scores_train.shape[1]*5
x, y = np.meshgrid(np.arange(1,width+1,5), np.arange(1,length+1))
fig = plt.figure()
ax1 = fig.add_subplot(1,2,1, projection='3d')
ax1.plot_surface(x, y, scores_train)
ax1.set_title("Training scores")
ax1.set_xlabel("Neurons")
ax1.set_ylabel("Layers")
ax2 = fig.add_subplot(1,2,2, projection='3d')
ax2.plot_surface(x, y, scores_test)
ax2.set_title("Test scores")
ax2.set_xlabel("Neurons")
ax2.set_ylabel("Layers")
plt.show()

# e) For dropout, we can't implement it using sklearn so we will use sknn.mlp  Classifier. This module can also be used for the tasks above. It offers more flexibility than sklearn.
# Before using it you might need to install it using: pip install scikit-neuralnetwork in command line
# If you get an error about downsampling run the following command: install --upgrade https://github.com/Lasagne/Lasagne/archive/master.zip
from sknn.mlp import Classifier, Layer

# Availble activation functions 'Rectifier', 'Sigmoid', 'Tanh', 'ExpLin'. for hidden layers and 'Linear', 'Softmax' for the output layer. This is a classification problem so we need to use Softmax
print("Trining a classifier with 4 hidden layers 10 neurons each... (ignore the warnings)")
clf = Classifier(layers=[Layer(type="Tanh", units=8),Layer(type="Tanh",units=10,dropout=0.3),
                         Layer(type="Tanh",units=10,dropout=0.3),Layer(type="Tanh",units=10,dropout=0.3),
                         Layer(type="Tanh",units=10,dropout=0.3), Layer(type="Softmax")],batch_size=200)
clf.fit(X_train,y_train)
print("Training Score",clf.score(X_train,y_train))
print("Test Score",clf.score(X_test,y_test))
# f) Back tp sklearn MLP classifier to try more activation functions
# Availble activation functions: {'identity', 'logistic', 'tanh', 'relu'}, default:'relu'
clf = MLPClassifier(solver='lbfgs',hidden_layer_sizes=(10,10),activation='relu')
clf.fit(X_test,y_test)
print("Training Score",clf.score(X_train,y_train))
print("Test Score",clf.score(X_test,y_test))





