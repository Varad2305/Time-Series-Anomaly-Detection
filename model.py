import sys,os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import LSTM,Dense,Bidirectional,Dropout,Flatten
from keras.layers import Conv1D,MaxPooling1D,GlobalAveragePooling1D

from utils import create_dataset,get_window_size
from utils import get_optimal_threshold,append_list_as_row

from sklearn.metrics import classification_report,roc_auc_score,mean_squared_error
from sklearn.preprocessing import MinMaxScaler	
from keras_self_attention import SeqSelfAttention


def create_model(window_size):
	"""
	Input : Lomb Scargle periodogram estimated window size
	Output : Proposed model in Keras
	"""
	model = Sequential()
	model.add(Bidirectional(LSTM(10,activation='tanh',input_shape=(window_size,1),return_sequences=True)))
	model.add(SeqSelfAttention(attention_activation='sigmoid',name='Attention'))
	model.add(Flatten())
	model.add(Dropout(0.5))
	model.add(Dense(1))
	model.compile(optimizer='adam',loss='mse')
	return model

def data(data_file_path):
	"""
	Input : Path to dataset
	Output : Data in the required format
	"""
	scaler = MinMaxScaler()
	data_sheet = pd.read_csv(data_file_path)
	_,window_size = get_window_size(data_sheet)
	features,values = create_dataset(data_sheet,window_size)
	features = scaler.fit_transform(features)
	values = scaler.fit_transform(values)
	features = features.reshape(features.shape[0],features.shape[1],1)
	train_size = (int)(0.6*features.shape[0])
	all_truths = np.asarray(data_sheet['label'])
	X_train = features[:train_size]
	y_train = values[:train_size]
	X_test = features[train_size:]
	y_test = values[train_size:]
	return X_train,y_train,X_test,y_test,window_size,train_size,all_truths

def threshold(X_test:pd.DataFrame,y_test:pd.DataFrame,all_truths:np.ndarray,window_size:int,train_size:int,model):
	"""
	Input : Test dataset, all labels, window size and training size
	Output : Precision, Recall, F-Score and AUC
	"""
	all_truths = np.reshape(all_truths,[all_truths.shape[0],1])
	train_truths = all_truths[window_size:train_size]
	train_truths = np.reshape(train_truths,[train_truths.shape[0],1])
	test_truths = all_truths[train_size:]
	test_truths = np.reshape(test_truths,[test_truths.shape[0],1])
	test_truths = test_truths[window_size:]
	train_scores = np.abs(model.predict(X_train) - y_train)
	# print(X_train.shape,y_train.shape)
	train_scores = train_scores[window_size:]
	train_scores_copy = np.copy(train_scores)
	threshold = get_optimal_threshold(train_truths,train_scores_copy,return_metrics=False)
	test_scores = np.abs(model.predict(X_test) - y_test)
	test_scores_copy = np.copy(test_scores)

	for i in range(test_scores.shape[0]):
		if(test_scores[i] > threshold):
			test_scores_copy[i] = 1
		else:
			test_scores_copy[i] = 0
		
	res = classification_report(test_truths,test_scores_copy,output_dict=True)
	try:
		auc = roc_auc_score(test_truths,test_scores_copy)
	except:
		auc = -1
	try:
		precision = res['1']['precision']
	except:
		precision = 0
	try:
		recall = res['1']['recall']
	except:
		recall = 0
	try:
		fscore = res['1']['f1-score']
	except:
		fscore = 0
	return precision,recall,fscore,auc


if __name__ == '__main__':
    directory = sys.argv[1]
    write_file = "model_results.csv"
    titles = ['Directory','Filename','Precision','Recall','F-Score','AUC']
    append_list_as_row(write_file,titles)

    for filename in os.listdir(directory):
    	print(filename)
    	X_train,y_train,X_test,y_test,window_size,train_size,all_truths = data(directory+filename)
    	model = create_model(window_size)
    	history = model.fit(X_train,y_train,epochs=100,batch_size=32,verbose=2)
    	o_prec,o_rec,o_fscore,o_auc = otsu_threshold(X_test,y_test,all_truths,window_size,train_size,model_1)
    	row = [os.path.dirname(directory),filename,o_prec,o_rec,o_fscore,o_auc]
    	append_list_as_row(write_file,row)