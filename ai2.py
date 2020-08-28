# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 11:56:14 2019

@author: pc
"""

#%%
#multiple linear regression
import pandas as pd

df=pd.read_csv('job2.csv')

x=df.iloc[:,[0,1]].values #it is standard numpy array we are taking all rows and 0. and 1. columns as coefficent
#x.shape when we do this x is appropriate for usage of sklearn ,it gives us (10,2)
y=df.income.values.reshape(-1,1)

multiple_linear_regression=LinearRegression()
multiple_linear_regression.fit(x,y)

print("bo: ",multiple_linear_regression.intercept_ )
print("b1: ",multiple_linear_regression.coef_ )

multiple_linear_regression.predict(np.array([[35,5],[35,10]]))
#we are looking for the new incomes of age:35 ,experience:5 ; age=35 , experience=10
#%%