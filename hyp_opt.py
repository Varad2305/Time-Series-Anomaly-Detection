import sys,os
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import Model,Sequential
from keras.layers import Conv1D,MaxPooling1D,Dropout,Dense,Bidirectional,LSTM
from keras.layers import Input,GlobalAveragePooling1D,Flatten
from keras_self_attention import SeqSelfAttention
from sklearn.preprocessing import MinMaxScaler
from utils import create_dataset,get_window_size
from hyperas.distributions import uniform,choice
from hyperas import optim
from hyperopt import Trials, STATUS_OK, tpe

def create_model(X_train,X_test,y_train,y_test):
    #Model begins
    model = Sequential()
    model.add(Input(shape=(window_size,1)))
    model.add(Bidirectional(LSTM({{choice([4,8,16,32])}},activation='tanh',input_shape=(window_size,1),return_sequences=True)))
    model.add(SeqSelfAttention(attention_activation='sigmoid',name='Attention'))
    model.add(Dropout({{uniform(0,1)}}))
    model.add(Dense(1))
    model.compile(optimizer='adam',loss='mse')
    # return model


    # model = Sequential()
    # model.add(Input(shape=(window_size,1)))
    # model.add(Bidirectional(LSTM({{choice([4,8,16,32])}},activation='tanh',input_shape=(window_size,1),return_sequences=True)))
    # model.add(SeqSelfAttention(attention_activation='sigmoid',name='Attention'))
    # # model.add(Flatten())
    # model.add(Dropout({{uniform(0,1)}}))
    # model.add(Dense(1))

    
    #Model ends

    #Compiling, fitting
    
    model.compile(loss='mse',optimizer='adam',metrics=['mse'])
    history = model.fit(X_train,y_train,batch_size=32,epochs=100,verbose=0,validation_split=0.1)
    validation_loss = np.amin(history.history['val_mse'])
    return {'loss': validation_loss, 'status': STATUS_OK, 'model': model}

def data():
    data_sheet = pd.read_csv(sys.argv[1])
    _,window_size = get_window_size(data_sheet)
    features,values = create_dataset(data_sheet,window_size)
    scaler = MinMaxScaler()
    train_size = (int)(features.shape[0]*0.6)
    X_train = features[:train_size]
    y_train = values[:train_size]
    X_test = features[train_size:]
    y_test = values[train_size:]
    return X_train,y_train,X_test,y_test

if __name__ == '__main__':
    best_run,best_model = optim.minimize(model=create_model,data=data,algo=tpe.suggest,max_evals=5,trials=Trials())
    X_train,y_train,X_test,y_test = data()
    print("Evaluation of best performing model:")
    print(best_model.evaluate(X_test, y_test,verbose=0))
    print("Best performing model chosen hyper-parameters:")
    print(best_run)