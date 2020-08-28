# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 16:06:03 2019

@author: pc
"""
#decision tree regression with python

#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df=pd.read_csv('decision.csv',sep=';',header=None)

x=df.iloc[:,0].values.reshape(-1,1)
y=df.iloc[:,1].values.reshape(-1,1)

#%%

from sklearn.tree import DecisionTreeRegressor
tree_reg=DecisionTreeRegressor()
tree_reg.fit(x,y)

#tree_reg.predict(5.5)
#x=np.arange(min(x),max(x),0.01).reshape(-1,1)
y_head=tree_reg.predict(x)

#%%

plt.scatter(x,y,color='red')
plt.plot(x,y_head,color='green')
plt.xlabel('tribun level')
plt.ylabel('price')
plt.show()

