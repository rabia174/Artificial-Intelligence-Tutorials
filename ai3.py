# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 13:50:09 2019

@author: pc
"""
#%%
import pandas as pd
import matplotlib.pyplot as plt

df=pd.read_csv('car.csv')
#if our file was separated by ;,then we should write
#df=pd.read_csv('car.csv',sep=';')

y=df.max_speed.values.reshape(-1,1)
x=df.price.values.reshape(-1,1)

plt.scatter(x,y)
plt.xlabel('price')
plt.ylabel('max speed')
plt.show()

#Linear regression  y=b0+b1*x
#multiple linear regression y=b0 + b1*x1 + b2*x2

#%%

from sklearn.linear_model import LinearRegression

lr=LinearRegression()

lr.fit(x,y)


#%%predict
y_head=lr.predict(x)

plt.scatter(x,y)
plt.plot(x,y_head,color='red',label='linear')
plt.show()
pr=lr.predict([[10000]])
pr
#this is not appropriate for linear regression model so we have add another coefficient
# y=b0+b1*x+b2*x^2+b3*x^3
#%%
#polynomial regression y=b0+b1*x+b2*x^2+...+bn*x^n
from sklearn.preprocessing import PolynomialFeatures
polynomial_regression=PolynomialFeatures(degree=2)
polynomial_regression2=PolynomialFeatures(degree=4)

x_polynomial=polynomial_regression.fit_transform(x)
x_polynomial2=polynomial_regression2.fit_transform(x)

#%% fit

linear_regression2=LinearRegression()
linear_regression2.fit(x_polynomial,y)

lr3=LinearRegression()
lr3.fit(x_polynomial2,y)

#%%
#slets see all of them together
y_head2=linear_regression2.predict( x_polynomial )
y_head3=lr3.predict( x_polynomial2)
plt.scatter(x,y)
plt.plot(x,y_head,color='red',label='linear')
plt.plot(x,y_head2,color='green',label='polynomial')
plt.plot(x,y_head3,color='black',label='polynomial degree=4')
plt.legend()
plt.show()