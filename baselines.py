# Imports
import pandas as pd
import numpy as np
import sys,os,time
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,roc_auc_score
from sklearn.preprocessing import MinMaxScaler
from DeepADoTS.src.algorithms import DAGMM,LSTMED,Donut
from DeepADoTS.src.algorithms import RecurrentEBM as REBM

from utils import *

#Init scaler
scaler = MinMaxScaler()

#Take command line args for directory to be run on and file to be written in
directory = sys.argv[1]
model_name = sys.argv[2]
write_file = "baselines_results.csv"

#Counter for current file number
counter = 1
total_files = len(os.listdir(directory))
print("TOTAL FILES: " + str(total_files))

total_start = time.time()

#Run on each file in the directory
for filename in os.listdir(directory):
	print(str(counter) + "/" + str(total_files))
	print("RUNNING ON FILE :" + filename)
	
	dataset = directory+filename

	#Load dataset
	data_sheet = pd.read_csv(dataset,parse_dates=True)

	#Get window size
	window_size = (int)(get_window_size(data_sheet))
	
	print("window_size: "+str(window_size))
	
	if model_name == 'LSTMED':
		model = LSTMED(sequence_length = window_size)
	
	if model_name == 'DAGMM':
		model = DAGMM(sequence_length = window_size)
	
	if model_name == 'REBM':
		model = REBM()
	
	if model_name == 'Donut':
		model = Donut()
	
	values = np.asarray(data_sheet['value'])
	values = np.reshape(values,[values.shape[0],1])
	values = pd.DataFrame(values)
	
	#Training. Calculate time taken to train
	start_time = time.time()
	history = model.fit(values)
	end_time = time.time()
	
	training_time = end_time - start_time
	scores = model.predict(values)
	
	truths = np.asarray(data_sheet['label'])
	truths = np.reshape(truths,[truths.shape[0],1])
	
	scores_std_dev = np.std(scores)
	scores_mean = np.mean(scores)
	scores_copy = np.copy(scores)
	
	#Get threshold
	threshold = get_optimal_threshold(truths,scores_copy,return_metrics=False)
	
	#Classify as anomaly or not
	for i in range(scores.shape[0]):
		if(scores[i] > threshold):
			scores_copy[i] = 1
		else:
			scores_copy[i] = 0
	
	#Get results
	res = classification_report(truths,scores_copy,output_dict=True)
	
	#Kept in a try catch block as sometimes calculating AUC gives error
	try:
		auc = roc_auc_score(truths,scores)
	except:
		auc = -1
	
	#Kept in a try catch block as sometimes 'res' might not have the key '1'.
	try:
		row = [os.path.dirname(directory),os.path.basename(dataset),model_name,threshold,res['1']['precision'],res['1']['recall'],res['1']['f1-score'],auc,str(training_time)]
		append_list_as_row(sys.argv[2],row)
	except:
		row = [os.path.dirname(directory),os.path.basename(dataset),model_name,threshold]
		append_list_as_row(sys.argv[2],row)
	counter = counter + 1

total_end = time.time()
_time = (total_end - total_start)/60
print("Phew that took " + str(_time) + " mins")
notification()