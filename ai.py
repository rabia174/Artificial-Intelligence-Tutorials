# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 10:03:19 2019

@author: pc
"""



#%%
import matplotlib.pyplot as plt
import pandas as pd
#if we import the dataset as array,skiprows=1
#then convert array into the dataframe
#df=pd.DataFrame( cesareancsv,columns=['regions','year-2002','year-2014','year-2015'])

#use it in your dataframe
#plt.scatter( df['year-2002'],df['year-2014'])
#plt.scatter( df['year-2014'],df['year-2015'])

#if we import it as dataframe we can directly use it
plt.scatter(jobcsv.experience,jobcsv.income)
plt.xlabel('experience')
plt.ylabel('income')
plt.show()
#%%

import matplotlib.pyplot as plt
import pandas as pd
from   sklearn.linear_model import LinearRegression

#initializing linear regression model
linear_reg=LinearRegression()
df=jobcsv
#if we say x=df.experience.values it gives us the shape (14,  it means 14 rows and 1 column but sklearn doesn't accept it
#to convert x and y to numpy array we use .values method
x=df.experience.values.reshape(-1,1)
y=df.income.values.reshape(-1,1)
#x=df.experience is in the type of pandas series so to convert it into numpy array use .values


#fitting line into the scatter plot
linear_reg.fit(x,y)

#our equation is in the form of y=ax+b, y=b0+b1*x  here b0 is coefficent or bias,intersection at yaxis,intercept
#and b1 is coefficient 

#there are two ways to find the b0

b0=linear_reg.predict([[0]])
print(b0)

bo_=linear_reg.intercept_
print(b0)

#these two methods above should give the same result
#and we can find the b1 which is coefficient by using coef_ method
b1=linear_reg.coef_
print(b1)

#and then now we can calculate the new income by using the formula   new_income=bo+b1*experience
#for example let's calculate the 20 years of experience
experience_=20
new_income=b0+b1*experience_
print( new_income )

#or we can calculate it by using library method predict

print(linear_reg.predict([[20]])) #predict method expects yu to enter values in 2d array form

#%%
#simple linear regression
import pandas as pd
import numpy as np
from   sklearn.linear_model import LinearRegression

#initializing linear regression model
linear_reg=LinearRegression()
df=jobcsv
#if we say x=df.experience.values it gives us the shape (14,  it means 14 rows and 1 column but sklearn doesn't accept it
#to convert x and y to numpy array we use .values method
x=df.experience.values.reshape(-1,1)
y=df.income.values.reshape(-1,1)
#x=df.experience is in the type of pandas series so to convert it into numpy array use .values


#fitting line into the scatter plot
linear_reg.fit(x,y)

array=np.arange(0,10)# if we do it 16 we'll get a longer line than the previous one
array.shape #it prints(16,  so to fix it do
array=array.reshape(-1,1)
array.shape #noe we've fixed it :) it gives (16,1)

plt.scatter(x,y)
plt.show()

y_head=linear_reg.predict(array)
plt.plot(array,y_head,color="red")
