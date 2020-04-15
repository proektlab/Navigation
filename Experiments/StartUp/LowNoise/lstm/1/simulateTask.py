import torch
from torch.utils.data import Dataset, DataLoader
import sys
import scipy.io
import numpy as np
import glob
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import os
import torch.nn.functional as F
import time
from torch.autograd import Function, Variable
import pickle
import shutil
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

from learnTask import NNNModel
from learnTask import NUM_NEURONS
from learnTask import NOISE_LEVEL
from learnTask import DEFAULT_NOISE_LEVEL
from buildData import buildData
from buildData import INPUT_NOISE
from buildData import DEFAULT_INPUT_NOISE

#%%
for MODEL_IDS in [-1]:
    
    USE_TEST_CASES = False
    USE_MODEL = MODEL_IDS #-1 for latest model
    
    inputData, targetData, classes = buildData(USE_TEST_CASES)
    
    experimentName = os.path.dirname(os.path.abspath(__file__))
    rootDir = '{}/../'.format(os.path.dirname(os.path.abspath(__file__)))
    experimentDir = '{}/'.format(experimentName)
    
    experimentScript = '{}learnTask.py'.format(experimentDir)
    experimentDataScript = '{}buildData.py'.format(experimentDir)
    
    #experimentParameters = '{}parameters.pkl'.format(experimentDir)
    #
    #if os.path.exists(experimentParameters):
    #    file = open(experimentParameters, 'rb')
    #    parameters = pickle.load(file)
    #    file.close()
        
    print('  Starting training...')
    
    if len(inputData.shape) == 3: # Trial data
        allInputs = torch.from_numpy(np.transpose(inputData, (2, 1, 0)))
        inputSize = inputData.shape[0]
    else: # Continous data
        allInputs = torch.from_numpy(inputData).unsqueeze(0)
        inputSize = inputData.shape[1]
    
    
    model = NNNModel(inputSize, int(np.max(targetData)+1), NUM_NEURONS)
    startEpoch = model.tryLoad('{}/savedModels/'.format(experimentDir), USE_MODEL)
    model.double()
    
    NUM_TRIALS = 1000
    
    #allInputs = torch.zeros(NUM_TRIALS, inputData.shape[1], inputData.shape[0])
    #allInputs = torch.from_numpy(np.transpose(inputData, (2, 1, 0)))
    
    
    predicted, dynamics= model(allInputs, train=False)
    x,taskDecision = predicted.max(2)
    
    noiseString = ''
    if not NOISE_LEVEL == DEFAULT_NOISE_LEVEL:
        noiseString = '_noise{}'.format( NOISE_LEVEL)
        
    inputString = ''
    if not INPUT_NOISE == DEFAULT_INPUT_NOISE:
        inputString = '_input{}'.format( INPUT_NOISE)
        
    modelString = ''
    if not USE_MODEL ==  -1:
        modelString = '_model{}'.format(USE_MODEL)
        
    
    if not USE_TEST_CASES:
        scipy.io.savemat('{}networkTrace{}{}{}.mat'.format(experimentDir, modelString, noiseString, inputString), {'inputs' : inputData, 'numTrials' : NUM_TRIALS, 'dynamics': dynamics, 'outputs' : taskDecision.detach().numpy(), 'targets' : targetData, 'classes' : classes})
    else:
        scipy.io.savemat('{}networkTraceTests{}{}{}.mat'.format(experimentDir, modelString, noiseString, inputString), {'inputs' : inputData, 'numTrials' : NUM_TRIALS, 'dynamics': dynamics, 'outputs' : taskDecision.detach().numpy(), 'targets' : targetData, 'classes' : classes})
    
    print('scp sharsnik@192.168.1.98:{}*.mat F:\Dropbox\ConservationOfAgentDynamics\ExperimentalData\{}\\'.format(experimentDir, experimentName))
    
    pca = PCA(n_components=3)
    pcaData = pca.fit_transform(dynamics[:,0,:])
    
    indices = range(0,80)
    
    fig = plt.figure(1)
    ax = Axes3D(fig)
    ax.plot3D(pcaData[indices,0],pcaData[indices,1],pcaData[indices,2])
    plt.draw()
    plt.pause(0.01)

