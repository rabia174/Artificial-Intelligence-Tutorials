# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 15:09:48 2019

@author: pc
"""
#%%Polynomial Linear Regression

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

#first I'll show you how to deal with linear regression

dataFrame=pd.read_csv('C:/Users/pc/Desktop/udemy/python/car.csv')

x=dataFrame.max_speed.values.reshape(-1,1)
y=dataFrame.price.values.reshape(-1,1)

linear_model=LinearRegression()
linear_model.fit(x,y)

y_head=linear_model.predict(x)
#it is enough to draw a simple linear regression line

#as the degree increases we get better prediction results
polynomial_regression_model=PolynomialFeatures(degree=10)
x_polynomial=polynomial_regression_model.fit_transform(x)

linear_model_poly=LinearRegression()
linear_model_poly.fit(x_polynomial,y)

y_head_poly=linear_model_poly.predict(x_polynomial)

plt.scatter(x,y)
plt.plot(x,y_head,color='red',label='linear')
plt.plot(x,y_head_poly,color='green',label='polynomial')
plt.legend()
plt.show()


