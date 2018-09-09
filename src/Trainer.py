import os
import numpy as np
import pandas as pd
import random
import time
import sys
from tqdm import tqdm

import torch
import torch.nn as nn
import torchvision
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F

import pyro
from pyro.distributions import Normal
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam

from Models import *
from DatasetGenerator import *


#--------------------------------------------------------------------------------
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Trainer ():

    def train (self, TrainPath, ValidPath, nnArchitecture, nnClassCount, trBatchSize, trMaxEpoch, timestampLaunch, checkpoint):

        #-------------------- START: INITIALIZATION
        start_epoch=0
        lossMIN = np.float('inf')


        #-------------------- SETTINGS: NETWORK ARCHITECTURE
        model = nnArchitecture['model'].to(device)


        #-------------------- SETTINGS: OPTIMIZER & SCHEDULER
        # TODO: pyro optimizer with ELBO and scheduler

        optimizer = 0
        scheduler = 0



        #-------------------- SETTINGS: LOSS
        loss = torch.nn.MSELoss()

        
        #-------------------- CHECKPOINT: LOAD

        if checkpoint != None:
            saved_parms=torch.load(checkpoint)
            model.load_state_dict(saved_parms['state_dict'])
            optimizer.load_state_dict(saved_parms['optimizer'])
            start_epoch= saved_parms['epochID']
            lossMIN    = saved_parms['best_loss']
            accMax     =saved_parms['best_acc']
            print ("****** Model Loaded ******")


        #--------------------- LOGS: TRAIN-MODEL LOGS
        sub = pd.DataFrame()
        timestamps = []
        archs = []
        losses = []


        for epochID in range (start_epoch, trMaxEpoch):

            #-------------------- SETTINGS: DATASET BUILDERS

            dataLoaderTrain = DatasetGenerator(TrainPath, transformSequence)
            dataLoaderVal =   DatasetGenerator(ValidPath, transformSequence)


            timestampTime = time.strftime("%H%M%S")
            timestampDate = time.strftime("%d%m%Y")
            timestampSTART = timestampDate + '-' + timestampTime

            print (str(epochID)+"/" + str(trMaxEpoch) + "---")

            self.epochTrain (model, dataLoaderTrain, optimizer, scheduler, trMaxEpoch, nnClassCount, loss, trBatchSize)
            lossVal, losstensor = self.epochVal (model, dataLoaderVal, optimizer, scheduler, trMaxEpoch, nnClassCount, loss, trBatchSize)


            timestampTime = time.strftime("%H%M%S")
            timestampDate = time.strftime("%d%m%Y")
            launchTimestamp = timestampDate + '-' + timestampTime

            scheduler.step(losstensor.item())

            if lossVal < lossMIN:
                lossMIN = lossVal

                timestamps.append(launchTimestamp)
                archs.append(nnArchitecture['name'])
                losses.append(lossVal)

                model_name = '../pyro/model-m-best_loss.pth.tar'

                states = {'epochID': epochID + 1,
                            'arch': nnArchitecture['name'],
                            'state_dict': model.state_dict(),
                            'best_loss':lossMIN,
                            'optimizer' : optimizer.state_dict()}

                torch.save(states, model_name)
                print ('Epoch [' + str(epochID + 1) + '] [save] [' + launchTimestamp + '] loss= ' + str(lossVal))

            else:
                print ('Epoch [' + str(epochID + 1) + '] [----] [' + launchTimestamp + '] loss= ' + str(lossVal))

        sub['timestamp'] = timestamps
        sub['archs'] = archs
        sub['loss'] = losses

        sub.to_csv('../pyro/' + nnArchitecture['name'] + '.csv', index=True)

    #--------------------------------------------------------------------------------
    def epochTrain (self, model, dataLoader, optimizer, scheduler, epochMax, classCount, loss, trBatchSize):

        phase='train'
        with torch.set_grad_enabled(phase == 'train'):
            for batchID, (input_, target_) in tqdm(enumerate (dataLoader)):
                # print (input_.size(), seg.size())
                target_ = target_.long()
                input_  = input_.float()

                varInputHigh = input_.to(device)
                varTarget    = target.to(device)

                varOutput = model(varInputHigh)
                lossvalue = loss(varOutput, varTarget)

                optimizer.zero_grad()
                lossvalue.backward()
                optimizer.step()

    #--------------------------------------------------------------------------------
    def epochVal (self, model, dataLoader, optimizer, scheduler, epochMax, classCount, loss, trBatchSize):

        model.eval ()

        lossVal = 0
        lossValNorm = 0
        losstensorMean = 0

        with torch.no_grad():
            for i, (input_, target_) in enumerate (dataLoader):

                target_ = target_.long()
                input_  = input_.float()

                varInputHigh = input_.to(device)
                varTarget    = target.to(device)

                varOutput  = model(varInputHigh)
                losstensor = loss(varOutput, varTarget)

                losstensorMean += losstensor
                lossVal += losstensor.item()
                lossValNorm += 1

            outLoss = lossVal / lossValNorm
            losstensorMean = losstensorMean / lossValNorm

        return outLoss, losstensorMean
