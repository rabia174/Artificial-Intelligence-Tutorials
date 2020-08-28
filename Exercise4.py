import pandas as pd
from keras.models import *
from keras.layers import *
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt


def series_to_supervised(data,  dropnan=True):
    n_vars = data.shape[1]
    names = [('var%d(t-%d)' % (j + 1, 1)) for j in range(n_vars)] +['var%d(t)' % 1]
    agg = pd.concat([data, data.ix[:, 0].shift(-1)], axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


## Data can be downloaded from: http://archive.ics.uci.edu/ml/machine-learning-databases/00235/
## Just open the zip file and grab the file 'household_power_consumption.txt' put it in the directory
## that you would like to run the code.

df = pd.read_csv('household_power_consumption.txt', sep=';',
                 parse_dates={'dt': ['Date', 'Time']}, infer_datetime_format=True,
                 low_memory=False, na_values=['nan', '?'], index_col='dt')
## resampling of data over hour
df = df.resample('h').mean()
print(df.shape)

Xy = series_to_supervised(df, dropnan=True)
Xy = Xy.values
scaler2 = MinMaxScaler(feature_range=(0, 1))
Xy=scaler2.fit_transform(Xy)
# print(Xy)
# df.drop(Xy.columns[0], 1, inplace=True)
X = np.array(Xy[:, :-1])
y = np.array(Xy[:, -1])

test_rate = 0.2
split = int(X.shape[0] * test_rate)
X_train = X[0:-split, :]
X_test = X[-split:, :]
y_train = y[0:-split]
y_test = y[-split:]

X_train = X_train.reshape((X_train.shape[0], 1, 7))
X_test = X_test.reshape((X_test.shape[0], 1, 7))

model = Sequential()
model.add(LSTM(100, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2))
#    model.add(LSTM(70))
#    model.add(Dropout(0.3))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

# fit network
history = model.fit(X_train, y_train, epochs=5, batch_size=70, validation_data=(X_test, y_test), verbose=2,shuffle=False)

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()

# make a prediction
yhat = model.predict(X_test)
X_test = X_test.reshape((X_test.shape[0], 7))
# invert scaling for forecast
inv_yhat = np.concatenate((X_test,yhat), axis=1)
inv_yhat = scaler2.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:, -1]
# invert scaling for actual
y_test = y_test.reshape((len(y_test), 1))
inv_y = np.concatenate((X_test,y_test), axis=1)
inv_y = scaler2.inverse_transform(inv_y)
inv_y = inv_y[:, 0]
# calculate RMSE
rmse = np.sqrt(np.mean((inv_y - inv_yhat) ** 2))
print('Test RMSE: %.3f' % rmse)

## time steps, every step is one hour (you can easily convert the time step to the actual time index)
## for a demonstration purpose, I only compare the predictions in 200 hours.

aa = [x for x in range(100)]
plt.plot(aa, inv_y[:100], marker='.', label="actual")
plt.plot(aa, inv_yhat[:100], 'r', label="prediction")
plt.ylabel('Global_active_power', size=15)
plt.xlabel('Time step', size=15)
plt.legend(fontsize=15)
plt.show()
