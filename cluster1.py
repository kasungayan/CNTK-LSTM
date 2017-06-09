import numpy as np
import pandas as pd
import random
import math as m

import sys
import os
import csv

from cntk.device import *
from cntk import Trainer
from cntk.layers import * 
from cntk.layers.typing import *
from cntk.learners import *
from cntk.ops import *
from cntk.logging import *
from cntk.metrics import *
from cntk.losses import *
from cntk.io import *

import cntk
import cntk.ops as o
import cntk.layers as l


from _cntk_py import set_fixed_random_seed, force_deterministic_algorithms

input_dim = 70
output_dim = 56
lstm_cell_dimension = 20

train_file_path = 'C:/local/cntk_new/cntk/NN5/cluster1/str_56_seasonali70.txt'
test_file_path = 'C:/local/cntk_new/cntk/NN5/cluster1/nn5_rsnob_1test.txt'
LSTM_USE_PEEPHOLES=True
BIAS=False


def create_train_data():

	listOfTuplesOfInputsLabels = []
	listOfInputs = []
	listOfLabels = []
	
	listOfTestInputs = []

	train_df = pd.read_csv(train_file_path , sep="|", header = None)
	
	for index, row in train_df.iterrows():
		input_df = train_df.values[index, 1]
		output_df = train_df.values[index, 2]

		input_split_df = input_df.split()
		output_df_df = output_df.split()

		input_array_df = np.asarray(input_split_df[1:len(input_split_df)], dtype=np.float32)
		output_array_df = np.asarray(output_df_df[1:len(output_df_df)], dtype=np.float32)

		tup=(input_array_df, output_array_df)
		listOfTuplesOfInputsLabels.append(tup)

	random.shuffle(listOfTuplesOfInputsLabels)
	
	for iseries in range(len(listOfTuplesOfInputsLabels)):
		series=listOfTuplesOfInputsLabels[iseries]
		listOfInputs.append(series[0])
		listOfLabels.append(series[1])
	
	listOfInputs= np.asarray(listOfInputs,dtype=np.float32)
	listOfLabels= np.asarray(listOfLabels,dtype=np.float32)
	
	test_df = pd.read_csv(test_file_path , sep="|", header = None)
	
	for index, row in test_df.iterrows():
		input_test_df = test_df.values[index, 1]
		input_split_test_df = input_test_df.split()

		input_test_array_df = np.asarray(input_split_test_df[1:len(input_split_test_df)], dtype=np.float32)
		listOfTestInputs.append(input_test_array_df)
	
	return listOfInputs, listOfLabels, listOfTestInputs
	
#def linear_layer_withNoBias(input_var, output_dim):
 # input_dim = input_var.shape[0]
  #times_param = o.parameter(shape=(input_dim, output_dim))
  #t=o.times(input_var, times_param)
  #return t
  
#def createLSTMnet2(input_var , num_hidden_layers1, hidden_layer_dim1, output_dim):
 # r=l.Recurrence(l.LSTM(hidden_layer_dim1, 
  #  use_peepholes=LSTM_USE_PEEPHOLES,init=glorot_uniform(seed=8888)))(input_var)
  #for i in range(1, num_hidden_layers1):
  #  r1=l.Recurrence(l.LSTM(hidden_layer_dim1, 
   #   use_peepholes=LSTM_USE_PEEPHOLES,init=glorot_uniform(seed=8888)))(r)
  #  r=r+r1    
  #return linear_layer_withNoBias(r, output_dim)

	
def train_model(features, labels,tests):

	gaussian_noise = 0.0004
	l2_regularization_weight = 0.0005
	minibatch_size = 128
	#test_minitbacth_size = 1
	max_epochs =10
	
	
	num_minibatches = len(features) // minibatch_size
	#num_test_minibatches = len(tests)//test_minitbacth_size
	epoch_size = len(features)*1
	
	feature = o.input_variable((input_dim),np.float32)
	label = o.input_variable((output_dim),np.float32)
	
	#Run the trainer on and perform model training
	num_passes = 1
	
	#feature = input((input_dim), np.float64)
    #label = input((ouput_dim), np.float64)
	
	#netout = createLSTMnet2(feature, 1, lstm_cell_dimension, output_dim)
	
	
	netout=Sequential([For(range(1), lambda i: Recurrence(LSTM(lstm_cell_dimension,use_peepholes=LSTM_USE_PEEPHOLES,init=glorot_uniform(seed=1)))),
                         Dense(output_dim,bias=BIAS,init=glorot_uniform(seed=1))])(feature)
	
	ce = squared_error(netout,label)
	pe = squared_error(netout, label)
	
	
	learner = momentum_sgd(netout.parameters, lr = learning_rate_schedule([(4,0.003),(16,0.002)], unit=UnitType.sample,epoch_size=epoch_size),
                           momentum=momentum_as_time_constant_schedule(minibatch_size / -m.log(0.9)), gaussian_noise_injection_std_dev = gaussian_noise,l2_regularization_weight =l2_regularization_weight)
	progress_printer = ProgressPrinter(1)
	trainer = Trainer(netout,(ce, pe), learner, progress_printer)
	
	tf = np.array_split(features,num_minibatches)
	tl = np.array_split(labels,num_minibatches)
	
	
	for epoch in range(max_epochs):	# loop over epochs
		for i in range(num_minibatches*num_passes): # multiply by the 
			features = np.ascontiguousarray(tf[i%num_minibatches])
			labels = np.ascontiguousarray(tl[i%num_minibatches])
			#features= np.array(features).tolist()
			#labels = np.array(labels).tolist()
			# Specify the mapping of input variables in the model to actual minibatch data to be trained with
			trainer.train_minibatch({feature : features, label : labels})
			progress_printer.update_with_trainer(trainer, with_metric=True) # log progress
		loss, metric, actual_samples = progress_printer.epoch_summary(with_metric=True)
		print(learner.learning_rate())
		
	#test_featues = np.array_split(tests,num_test_minibatches)	
	
	test_output=trainer.model.eval({feature: tests})
	#np.asarray(test_output,dtype=np.float32)
	#np.savetxt("model.txt", test_output, delimiter=" ",fmt='%f')
	#test_output.save_model("ciftrend_4.z")
	#test_output=np.array(test_output).tolist()
	#df = pd.DataFrame(test_output)
	#df.to_csv("model.csv")
	
	#with open("test.csv", "w") as output:
		#writer = csv.writer(output, lineterminator='\n')
		#writer.writerows(test_output)
		
	return test_output
	
if __name__ == '__main__':

	set_default_device(cpu())
	np.random.seed(1)
	random.seed(1)
	cntk.cntk_py.set_fixed_random_seed(1)
	cntk.cntk_py.force_deterministic_algorithms()
	#force_deterministic_algorithms(true)

	listOfTestInput = []
    # Specify the target device to be used for computing, if you do not want to
    # use the best available one, e.g.
    # try_set_default_device(cpu())
	features, labels,tests = create_train_data()
	#print(features)
	#print(labels)
	testshape = train_model(features, labels,tests)
	
	#print(shape[0])
	for il in range(len(testshape)):
		oneTestOut=testshape[il]
		listvalue = oneTestOut[oneTestOut.shape[0]-1,]
		listvalue= np.array(listvalue).tolist()
		listOfTestInput.append(listvalue)
	
	with open("forecastingcluster1.txt", "w") as output:
		writer = csv.writer(output, lineterminator='\n')
		writer.writerows(listOfTestInput)