import os
import numpy as np
import torch
import torch.nn as nn

import pyro
from pyro.distributions import Normal
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam

from Models import *
from Trainer import *
from DataGenerator import *

Trainer = Trainer()


#--------------------------------------------------------------------------------

def main (nnClassCount=nclasses):
	# "Define Architectures and run one by one"

	nnArchitectureList = [
							{
								'name': 'BN1',
								'model' : BayesianMLP(),
								'ckpt' : None
							}

						]

	for nnArchitecture in nnArchitectureList:
		runTrain(nnArchitecture=nnArchitecture)



def getDataPaths(path):
 	data = pd.read_csv(path)
 	imgpaths = data['Paths'].as_matrix()
 	return imgpaths

#--------------------------------------------------------------------------------

def runTrain(nnArchitecture = None):

	timestampTime = time.strftime("%H%M%S")
	timestampDate = time.strftime("%d%m%Y")
	timestampLaunch = timestampDate + '-' + timestampTime

	TrainPath = nnArchitecture['TrainPath']
	ValidPath = nnArchitecture['ValidPath']

	nnClassCount = nclasses

	#---- Training settings: batch size, maximum number of epochs
	trBatchSize = 4
	trMaxEpoch = 60*4

	print ('Training NN architecture = ', nnArchitecture['name'])
	info_dict = {
				'batch_size': trBatchSize,
				'architecture':nnArchitecture['name'] ,
				'number of epochs':trMaxEpoch,
				'train path':TrainPath, 
				'valid_path':ValidPath,
				'number of classes':nclasses,
				'Date-Time':	timestampLaunch
	} 
	if not os.path.exists('../modelsclaheWC11'): os.mkdir('../modelsclaheWC11')
	with open('../modelsclaheWC11/config.txt','w') as outFile:
		json.dump(info_dict, outFile)
	

	Trainer.train(TrainPath,  ValidPath, nnArchitecture, nnClassCount, trBatchSize, trMaxEpoch, timestampLaunch, nnArchitecture['ckpt'])


#--------------------------------------------------------------------------------

if __name__ == '__main__':
	main()
	# runTest()