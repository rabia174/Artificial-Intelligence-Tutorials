# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 17:08:47 2019

@author: pc
"""
#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df=pd.read_csv('decision.csv',sep=';',header=None)
x=df.iloc[:,0].values.reshape(-1,1)
y=df.iloc[:,1].values.reshape(-1,1)

#%%

from sklearn.ensemble import RandomForestRegressor
rf=RandomForestRegressor(n_estimators=100, random_state=42 )
rf.fit(x,y)

print('7.8 level price: ',rf.predict([[7.8]]))

x_=np.arange(min(x),max(x),0.01).reshape(-1,1)
y_head=rf.predict(x_)

plt.scatter(x,y,color='red' )
plt.plot(x_,y_head,color='green')
plt.xlabel('tribun level')
plt.ylabel('price')
plt.show()
