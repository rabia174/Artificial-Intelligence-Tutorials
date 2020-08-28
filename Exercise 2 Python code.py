import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
# read the data set
df = pd.read_csv('breast-cancer-wisconsin.data.txt')

# Replace the missing or non numeric values
df.dropna(inplace=True)
df=df._get_numeric_data()



# Drop the ID column
df.drop(['id'], 1, inplace=True)
# Create features and labels arrays
X = np.array(df.drop(['class'], 1))
y = np.array(df['class'])
# Scale the features array X
# The scaling of the data is important for convergence of the model
# There are different types of scaling
# scaler1 = MinMaxScaler(feature_range=(0,1))
# scaler1.fit(X)
# X = scaler1.transform(X)

scaler2 = StandardScaler()
scaler2.fit(X)
X = scaler2.transform(X)

# Transform the feature array y to a binary array 0 or 1
y = np.array(y==4).astype(int)

# Split the arrays into training and test arrays
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

print(X_train,X_test, y_train, y_test)