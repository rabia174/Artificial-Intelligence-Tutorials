
"""
Created on Sun Oct  6 00:47:45 2019
This code is created by one of this class students.
@author: rabia
"""
# Self Organizing Map

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('breast-cancer-wisconsin.data.txt')
dataset.replace('?','0',inplace=True )
dataset.rename(columns={'1000025': 'id'}, inplace=True)

dataset.drop('id',axis=1,inplace=True)
X = dataset.iloc[:, :10 ].values
y = dataset.iloc[:, 9:10 ].values.reshape(-1,)
y = np.array(y==4).astype(int)


# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
X = sc.fit_transform(X)

# Training the SOM
from minisom import MiniSom
som = MiniSom(x = 15, y = 15, input_len = 10, sigma = 1.8, learning_rate = 0.5)
som.random_weights_init(X)
som.train_batch(data = X, num_iteration = 500)

# Visualizing the results

from pylab import bone, pcolor, colorbar, plot, show
bone()
pcolor(som.distance_map().T)
colorbar()

markers = [ '+','o' ]
colors =  [ 'g','r' ]

for i, x in enumerate(X):
    w = som.winner(x)
    plot(w[0] +.5,
         w[1] +.5,
         markers[y[i]],
         markeredgecolor = colors[y[i]],
         markerfacecolor = 'None',
         markersize = 10,
         markeredgewidth = 2)
show()
