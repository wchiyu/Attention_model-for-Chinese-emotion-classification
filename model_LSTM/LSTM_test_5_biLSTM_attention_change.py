from keras.layers import Bidirectional, Concatenate, Permute, Dot, Input, LSTM, Multiply
from keras.layers import RepeatVector, Dense, Activation, Lambda,Flatten,MaxPooling2D,Convolution2D
from keras.layers.core import Reshape,Permute
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.models import load_model, Model
from keras.layers.embeddings import Embedding
import keras.backend as K
import numpy as np
import sys
sys.setrecursionlimit(1000000)

X=np.load('all_data_one.npy')
Y=np.load('all_label_one.npy')

all_data=X.astype(np.int64)
#all_label=Y.astype(np.int32)

X_train=X[0:19000,:]
Y_train=Y[0:19000,:]
X_test=X[19000:,:]
Y_test=Y[19000:,:]

m_train=X_train.shape[0]
m_test=X_test.shape[0]
#parameters of embedding layer
input_dim_embedding=14322
out_dim_embedding=100
max_input_length=200
#parameters of LSTM layer
out_dim_LSTM=100
#parameter of output layer
output_dim=1
#timestep
Tx=200

# defined pre_LSTM
repeator = RepeatVector(Tx,name='rep')
concatenator = Concatenate(axis=-1,name='con')
densor1 = Dense(10, activation = "tanh",name='den1')
densor2 = Dense(1, activation = "relu",name='den2')
activator = Activation('softmax', name='attention_weights') # We are using a custom softmax(axis = 1) loaded in this notebook
dotor = Dot(axes = 1,name='dot')
permute_2_1=Permute((2,1))
#define post_LSTM
post_activation_LSTM_cell = LSTM(out_dim_LSTM, return_state = True)
reshape_post_lstm=Reshape((1,out_dim_LSTM))
concatenator_post_lstm=Concatenate(axis=1,name='con_post_lstm')
densor3=Dense(1)
densor4=Dense(output_dim)
output_layer=Activation('sigmoid')
#define cnn_layer:
s1=max_input_length
stride=1


#caculate the weights of attention model
def  one_step_attention(a,s_prev):
    s_prev=repeator(s_prev)
    concat=concatenator([a,s_prev])
    e=densor1(concat)
    energies = densor2(e)
    energies=permute_2_1(energies)
    alphas=activator(energies)
    alphas=permute_2_1(alphas)
    context=dotor([alphas,a])
    return context

#build the model
X=Input(shape=(max_input_length,))
s0 = Input(shape=(out_dim_LSTM,), name='s0')
c0 = Input(shape=(out_dim_LSTM,), name='c0')
s = s0
c = c0
s_list=[]
embedding=Embedding(input_dim=input_dim_embedding,output_dim=out_dim_embedding,input_length=max_input_length)(X)
a=Bidirectional(LSTM(units=out_dim_LSTM,return_sequences=True))(embedding)
for t in range(Tx):
    context=one_step_attention(a,s)
    s,_,c=post_activation_LSTM_cell(context,initial_state = [s,c])
    s_reshape=reshape_post_lstm(s)
    s_list.append(s_reshape)
cnn_input=s_list[0]
for i in range(1,len(s_list)):
    cnn_input=concatenator_post_lstm([cnn_input,s_list[i]])
#cnn_input_reshape=Reshape((300,100,1))(cnn_input)
#pool1 = MaxPooling2D(pool_size=(1,100),strides=stride,padding='valid',data_format='channels_last')(cnn_input_reshape)
#convo1=Convolution2D(filters=1,kernel_size=(1,100),strides=stride,padding='valid',data_format='channels_last',activation="relu")(cnn_input_reshape)
convo1=densor3(cnn_input)
flatten1=Flatten()(convo1)
outputs=densor4(flatten1)
outputs=output_layer(outputs)

model=Model(inputs=[X,s0,c0],outputs=outputs)
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

s0_train = np.zeros((m_train, out_dim_LSTM))
c0_train = np.zeros((m_train, out_dim_LSTM))
s0_test = np.zeros((m_test, out_dim_LSTM))
c0_test = np.zeros((m_test, out_dim_LSTM))
#model.summary()
history=model.fit([X_train,s0_train,c0_train],Y_train,validation_data=([X_test,s0_test,c0_test],Y_test),epochs=10,batch_size=128)

#loss=np.array(history.history['loss'])
#np.save('test_4_loss.npy',loss)


'''
acc=np.array(history.history['acc'])
val_acc=np.array(history.history['val_acc'])
loss=np.array(history.history['loss'])
val_loss=np.array(history.history['val_loss'])
np.save('test_4_acc.npy',acc)
np.save('test_4_val_acc.npy',val_acc)
np.save('test_4_loss.npy',loss)
np.save('test_4_val_loss.npy',val_loss)

model.save('LSTM_test_4.h5')
'''


