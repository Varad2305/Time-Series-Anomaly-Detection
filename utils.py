# Imports
import sys
import pandas as pd
import numpy as np
from scipy.signal import lombscargle
from datetime import timedelta
from dateutil.parser import parse
from csv import writer
from sklearn.metrics import accuracy_score, fbeta_score
from sklearn.metrics import precision_recall_fscore_support as prf
from scipy.signal import find_peaks


def create_dataset(data_sheet:pd.DataFrame,window_size:int):
	"""
	Input: Dataset in the form of a panda and window_size
	Output: A transormed dataset with window_size-1 features and 1 target value
	"""
	x = np.asarray(data_sheet['value'])
	x = np.reshape(x,[x.shape[0],1])
	aux1 = np.empty([1,window_size])
	aux2 = np.empty(1)

	for row in range(x.shape[0]-window_size):
		aux1 = np.append(aux1,x[row:row+window_size].T,axis=0)
		aux2 = np.append(aux2,x[row+window_size])

	features = np.delete(aux1,0,axis=0)
	aux2 = np.delete(aux2,0)
	values = np.reshape(aux2,[aux2.shape[0],1])
	return features,values


def get_window_size(data_sheet:pd.DataFrame):
	"""
	Input: Dataset in the form of a Panda Dataframe
	Output: Optimal window size estimated by the Lomb Scargle method
	"""
	times = np.asarray(data_sheet['timestamp'])
	ts = np.empty(0)

	for time in times:
		X = parse(time,fuzzy=True)
		td = timedelta(days =  X.day,hours = X.hour,minutes = X.minute,seconds = X.second)
		ts = np.append(ts,td.total_seconds())

	values = np.asarray(data_sheet['value'])
	f = np.linspace(0.01,100,100).T
	pgram = lombscargle(ts,values,f.T)
	peaks,_ = find_peaks(pgram)
	window_size = np.argmax(pgram)
	return peaks,window_size

def append_list_as_row(file_name,list_of_elem):
	"""
	Input : Filename and a list of data
	Function : Appends the list as a row in the file given by the filename
	"""
	with open(file_name,'a+',newline='') as write_obj:
		csv_writer = writer(write_obj)
		csv_writer.writerow(list_of_elem)

def get_optimal_threshold(y_test: list, score: list, steps=100, return_metrics=False):
	"""
	Input : Test actual values, all anomaly scores
	Output : Returns the optimum threshold after trying out 'steps' number of values between the minimum score and maximum score
	"""

	maximum = np.nanmax(score)
	minimum = np.nanmin(score)
	threshold = np.linspace(minimum, maximum, steps)
	metrics = list(get_metrics_by_thresholds(y_test, score, threshold))
	metrics = np.array(metrics).T
	anomalies, acc, prec, rec, f_score, f01_score = metrics
	if return_metrics:
		return anomalies, acc, prec, rec, f_score, f01_score, threshold
	else:
		return threshold[np.argmax(f_score)]

def get_metrics_by_thresholds(y_test: list, score: list, thresholds: list):
	"""
	Input : Test actual values, all anomaly scores, list of thresholds to be testes
	Output : Yields performance metrics for each threshold in the list
	"""
	
	for threshold in thresholds:
		anomaly = binarize(score, threshold=threshold)
		metrics = get_accuracy_precision_recall_fscore(y_test, anomaly)
		yield (anomaly.sum(), *metrics)      

def binarize(score, threshold=None):
	"""
	Input : All scores and threshold
	Output : Returns indices of points that are anomalous
	"""
	threshold = threshold if threshold is not None else threshold(score)
	score = np.where(np.isnan(score), np.nanmin(score) - sys.float_info.epsilon, score)
	return np.where(score >= threshold, 1, 0)

def get_accuracy_precision_recall_fscore(y_true: list, y_pred: list):
	"""
	Input : Actual labels and Predicted labels
	Output : Returns performance metrics
	"""
	accuracy = accuracy_score(y_true, y_pred)
	precision, recall, f_score, _ = prf(y_true, y_pred, average='binary', warn_for=())
	if precision == 0 and recall == 0:
		f01_score = 0	
	else:
		f01_score = fbeta_score(y_true, y_pred, average='binary', beta=0.1)
	return accuracy, precision, recall, f_score, f01_score      

def threshold(score):
	return np.nanmean(score) + 2 * np.nanstd(score)  