from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
import numpy as np

X=np.load('all_data_one.npy')
Y=np.load('all_label_one.npy')

all_data=X.astype(np.int64)
#all_label=Y.astype(np.int32)

X_train=X[0:19000,:]
Y_train=Y[0:19000,:]
X_test=X[19000:,:]
Y_test=Y[19000:,:]

#parameters of embedding layer
input_dim_embedding=14322
out_dim_embedding=100
max_input_length=200
#parameters of LSTM layer
out_dim_LSTM=100
#parameter of fc layer
out_dim_fc=100
#parameter of output layer
output_dim=1

model=Sequential()
model.add(Embedding(input_dim=input_dim_embedding,output_dim=out_dim_embedding,input_length=max_input_length))
#model.add(LSTM(units=out_dim_LSTM,return_sequences=True))
model.add(LSTM(output_dim=out_dim_LSTM,return_sequences=False))
model.add(Dense(output_dim))
model.add(Activation('sigmoid'))
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
history = model.fit(X_train, Y_train,validation_data=(X_test,Y_test),epochs=40, batch_size=128)

#model.save('LSTM_test_1.h5')

