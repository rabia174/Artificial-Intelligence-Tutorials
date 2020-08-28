from Data_preprocessing import X_train, y_train, X_test, y_test
from keras.models import *
from keras.layers import *
import numpy as np


l_input = Input(batch_shape=(None, X_train.shape[1]))
l_dense1 = Dense(10, activation='relu')(l_input)
dropout1 = Dropout(0.3)(l_dense1)
l_dense2 = Dense(10, activation='relu')(dropout1)
dropout2 = Dropout(0.3)(l_dense2)
out = Dense(2, activation='softmax')(dropout2)

model = Model(inputs=[l_input], outputs=[out])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
y_train = np.concatenate((y_train.reshape(-1,1)==1, y_train.reshape(-1,1)==0),axis=1)
y_test = np.concatenate((y_test.reshape(-1,1)==1, y_test.reshape(-1,1)==0),axis=1)
model.fit(X_train,y_train, epochs=100)

model.predict(X_test)

print("Training Loss/Accuracy:",model.evaluate(X_train,y_train))
print("Test Loss/Accuracy:", model.evaluate(X_test,y_test))
