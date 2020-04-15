import torch
from torch.utils.data import Dataset, DataLoader
import sys
import scipy.io
import numpy as np
import glob
import matplotlib
matplotlib.use("TkAgg")
#matplotlib.use("qt5agg")
import matplotlib.pyplot as plt
import os
import torch.nn.functional as F
import time
from torch.autograd import Function, Variable
import pickle
import shutil
from buildData import buildData

#%%

#matplotlib.interactive(True)



NUM_NEURONS = 100
DEFAULT_NOISE_LEVEL = 0.1
NOISE_LEVEL = DEFAULT_NOISE_LEVEL
USE_GRU = 0
DISPLAY = 1

NUM_SEQUENCES = 5000;
SEQUENCE_LENGTH = 200;

EXP_NAME = 'FlipFlop1Frame/LowNoiseNoInput'
MIN_ACCURACY = 1
ACCURACY_POWER = 1
MAX_LOSS = 0.05
LOSS_POWER = 2
MIN_EPOCH = 300
EPOCH_POWER = 1
WEIGHT_DECAY = 0.0;
MODULE_COUNT = 2;
MODULE_CONNECTEDNESS = 0.001;
LABELS = ['Loss', 'Accuracy']
LINE_COLORS = ['#0071bc', '#d85218', '#ecb01f', '#7d2e8d', '#76ab2f', '#4cbded', '#a1132e']

#sys.path.append('/data/connor/Dropbox/MachineLearningPipeline/')

        
class NNNModel(torch.nn.Module):
    def __init__(self, inputSize, numClasses, hiddenNeurons = 10, modules = 1, connectedness = 1):
        super(NNNModel, self).__init__()

        self.hiddenNeurons = hiddenNeurons
        
        self.modelConnections = torch.nn.Parameter()
        self.modelConnections.data = torch.zeros(hiddenNeurons, hiddenNeurons, requires_grad=False)
        self.modelConnections.requires_grad = False
        
        neuronsPerModule = int(hiddenNeurons/modules)
        
        totalConnections = np.ceil(neuronsPerModule**2 * connectedness)
        
        for mi in range(0, modules):
            for mj in range(0, modules):
                if mi  == mj:
                    for i in range(0, neuronsPerModule):
                        for j in range(0, neuronsPerModule):
                            self.modelConnections[mi*neuronsPerModule+i,mj*neuronsPerModule+j] = 1
                else:
                    counter = 0
                    connections = np.random.choice(int(neuronsPerModule**2), int(totalConnections), False)
                    #connections = range(0, int(totalConnections))
                    for i in range(0, neuronsPerModule):
                        for j in range(0, neuronsPerModule):
                            if np.isin(counter, connections):
                                self.modelConnections[mi*neuronsPerModule+i,mj*neuronsPerModule+j] = 1
                            counter += 1
                            
        #self.inputLayer = torch.nn.Linear(inputSize, hiddenNeurons)

        if USE_GRU:
            self.lstmLayer = torch.nn.GRUCell(inputSize, hiddenNeurons)
        else:
            self.lstmLayer = torch.nn.LSTMCell(inputSize, hiddenNeurons)
            
        self.outputLayer = torch.nn.Linear(hiddenNeurons, hiddenNeurons)
        self.finalLayer = torch.nn.Linear(hiddenNeurons, numClasses)
        self.softmax = torch.nn.LogSoftmax(dim=2)
        
        self.sigmoid = torch.nn.Sigmoid()
        
        self.apply(self.initWeights)
            
        
    def initWeights(self, parameter):
        mask = torch.cat((self.modelConnections, self.modelConnections, self.modelConnections, self.modelConnections))
        
        if isinstance(parameter, torch.nn.LSTMCell):
            parameter.weight_hh.data = parameter.weight_hh.data * mask
            
    def setGrad(self):
        mask = torch.cat((self.modelConnections, self.modelConnections, self.modelConnections, self.modelConnections))
                
        self.lstmLayer.weight_hh.grad = self.lstmLayer.weight_hh.grad * mask

    def forward(self, inputs, train=True):
        hiddenState = torch.zeros(inputs.shape[0], self.hiddenNeurons, dtype=torch.double)
        cellState = torch.zeros(inputs.shape[0], self.hiddenNeurons, dtype=torch.double)
        hiddenNoise = torch.zeros(inputs.shape[1],inputs.shape[0],self.hiddenNeurons, requires_grad=False, dtype=torch.double)
        cellNoise = torch.zeros(inputs.shape[1],inputs.shape[0],self.hiddenNeurons, requires_grad=False, dtype=torch.double)
        output = torch.zeros(inputs.shape[0], inputs.shape[1], self.hiddenNeurons, dtype=torch.double)

        #inputs = F.elu(self.inputLayer(inputs))

        hiddenNoise = hiddenNoise.data.normal_(0, std=NOISE_LEVEL)
        cellNoise = cellNoise.data.normal_(0, std=NOISE_LEVEL)

        inputs = inputs.permute(1,0,2)

        dynamics = []
        for t in range(inputs.shape[0]):
            if USE_GRU:
                hiddenState = self.lstmLayer(inputs[t], (hiddenState))
                hiddenState = hiddenState + hiddenNoise[t]
                
                output[:,t,:] = (hiddenState)
                if not train:
                    dynamics.append(hiddenState.detach().numpy())
            else:
                hiddenState, cellState = self.lstmLayer(inputs[t], (hiddenState, cellState))
                hiddenState = hiddenState + hiddenNoise[t]
                cellState = cellState + cellNoise[t]
                
                output[:,t,:] = (hiddenState)
                if not train:
                    dynamics.append(np.concatenate((hiddenState.detach().numpy(), cellState.detach().numpy()), 1))
               
        if not train:
            dynamics = np.stack(dynamics)
        
        taskOutput = self.finalLayer(output)
        taskOutput = self.softmax(taskOutput)

        return taskOutput, dynamics

    def tryLoad(self, backupDirectory, loadModel = -1):
        self.backupDirectory = backupDirectory
        try:
            os.mkdir(self.backupDirectory)
        except:
            pass

        paths = glob.glob(self.backupDirectory + '*.tar');
        epochNumber = 0
        if len(paths) > 0:
            ckpts = [int(s.split('.')[-2]) for s in paths]
            ix = np.argmax(ckpts)
            epochNumber = ckpts[ix]

        if loadModel > -1 and epochNumber > loadModel:
            if loadModel in ckpts:
                epochNumber = loadModel
                ix = ckpts.index(loadModel)

        if epochNumber > 0:
            self.load_state_dict(torch.load(paths[ix]))
            print("  Loaded model: {}".format(paths[ix]))

        return epochNumber

    def saveBackup(self, epochNumber):
        torch.save(self.state_dict(), self.backupDirectory + 'model.{:.0f}.tar'.format(epochNumber))

#def loss(predicted, target, weight):
#    return ((predicted - target) ** 2 * weight).mean()

def loss(predicted, targets, classes, plotPanel=None):    
    
    
    finalValues = predicted.view(-1,predicted.shape[2])
    targetValues = torch.squeeze(targets.view(-1, 1)).long()
    
    x,predictedClasses = finalValues.max(1)
    targetDecision = torch.squeeze(targetValues)
    
    weights = torch.zeros(predicted.shape[2], dtype=torch.double)
    accuracies = torch.zeros(predicted.shape[2])
    for i in range(0, predicted.shape[2]):
        weights[i] = torch.sum(targetValues == i)
        
        targetTimes = torch.squeeze(targetValues == i)
        accuracies[i] = torch.sum(predictedClasses[targetTimes] == targetDecision[targetTimes].long()).float() / torch.sum(targetTimes)
        
    weights = 1 / weights
    weights = weights / torch.sum(weights)
    
    finalLoss = torch.nn.NLLLoss(weight=weights)
    
    taskScore = finalLoss(finalValues, targetValues)
    
    
        
    targetTimes = torch.squeeze(torch.t(targetValues.nonzero()))
    
    x,taskDecision = finalValues.max(1)
    targetDecision = targetValues[targetTimes]
    targetDecision = torch.squeeze(targetDecision)

    accuracy = accuracies.mean()   
    
    if not plotPanel is None:
        #x,taskDecision = predicted.max(2)
        #taskDecision,y = taskDecision.max(1)
        #targetDecision,y = targets.max(1)
        #targetDecision = torch.squeeze(targetDecision)
        
        plotPanel.clear()
        line1, = plotPanel.plot(accuracies, 'go', label='Accuracy by class')   
        #line2, = plotPanel.plot((taskDecision.detach().numpy() + 1), 'b-', linewidth=0.5, label='Predicted decision')   #        
        #line3, = plotPanel.plot((targetDecision.detach().numpy() + 1), 'r-', linewidth=0.5, label='Target decision') 
        
        handles = []
        handles.append(line1)
        #handles.append(line2)
        #handles.append(line3)

        plotPanel.legend(handles=handles)
        
    return taskScore, accuracy

class WeightedSequences(Dataset):
    def initTrials(self, inputData, targetData, classes):
        self.x = torch.zeros(inputData.shape[2], inputData.shape[1], inputData.shape[0], requires_grad=False)
        self.y = torch.zeros(inputData.shape[2], targetData.shape[1], targetData.shape[0], requires_grad=False)
        self.classes = torch.zeros(inputData.shape[2], requires_grad=False)
        
        self.x = torch.from_numpy(np.transpose(inputData, (2, 1, 0)))
        self.y = torch.from_numpy(np.transpose(targetData, (2, 1, 0)))
        self.classes = torch.from_numpy(classes)

        self.len = inputData.shape[2]
        
    def initContinous(self, inputData, targetData, classData, numSequences, sequenceLength):
        self.x = torch.zeros(numSequences, sequenceLength, inputData.shape[1], requires_grad=False)
        self.y = torch.zeros(numSequences, sequenceLength, targetData.shape[1], requires_grad=False)
        self.classes = torch.zeros(numSequences, requires_grad=False)
        
        uniqueClasses = np.unique(classData)
        classIndices = []
        elementCounts = []
        for i in range(0, len(uniqueClasses)):
            putativeIndices = np.array(np.nonzero(classData[:,0] == uniqueClasses[i]))
            putativeIndices = np.delete(putativeIndices, np.nonzero(putativeIndices < sequenceLength)) 
            classIndices.append(putativeIndices)
            elementCounts.append(classIndices[i].shape[0])
            
        minTrials = int(np.ceil(numSequences / len(uniqueClasses)))
        minElementSize = min(elementCounts)
        
        if minTrials > minElementSize:
            minTrials = minElementSize # FIX ME to pad smaller trials
        
        allIndices = np.zeros([minTrials*len(uniqueClasses)])
        for i in range(0, len(uniqueClasses)):
            startTimes = np.random.choice(classIndices[i], minTrials, replace = False)
            allIndices[(0 + i*minTrials):(minTrials + i*minTrials)] = startTimes

        for i in range(0, len(allIndices)):
            endTime = int(allIndices[i]) + 1
            startTime = endTime - sequenceLength

            self.x[i, :, :] = torch.from_numpy(inputData[startTime:endTime, :])
            self.y[i, :, :] = torch.from_numpy(targetData[startTime:endTime, :])
            self.classes[i] = classData[endTime-1,0]

        self.len = numSequences

    def __getitem__(self, index):
        return self.x[index,:,:], self.y[index,:,:], self.classes[index]

    def __len__(self):
        return self.len

def loadMatlabData(filename):
    loadData = scipy.io.loadmat(filename, mat_dtype=False)
    processedData = dict()
    for key in loadData.keys():
        if (key[0] != '_' and key[1] != '_'):
            processedData[key] = loadData[key]
            if processedData[key].dtype.type is np.str_:
                processedData[key] = str(processedData[key])[2:-2]

    return processedData

class TrainingPlot:
    LOCAL_WINDOW = 10

    def __init__(self, filename, epochNumber, labels):
        self.epoch = []
        self.labels = labels
        self.saveFile = filename
        self.numLabels = len(labels)
        
        self.variables = [0 for i in range(self.numLabels)]
        for i in range(0, self.numLabels):
            self.variables[i] = []

        paths = glob.glob(self.saveFile)
        if len(paths) > 0:
            trainingData = scipy.io.loadmat(paths[0])
            self.epoch = (trainingData['x'][0]).tolist()
            for i in range(0, self.numLabels):
                self.variables[i] = (trainingData[self.labels[i]][0]).tolist()

            if self.epoch[-1] >= epochNumber:
                epochArray = np.array(self.epoch)
                epochArray = np.where(epochArray >= epochNumber)
                firstIndex = epochArray[0][0]

                if firstIndex > 0:
                    self.epoch = self.epoch[0:firstIndex - 1]
                    for i in range(0, self.numLabels):
                        self.variables[i] = self.variables[i][0:firstIndex - 1]
                else:
                    self.epoch = []
                    for i in range(0, self.numLabels):
                        self.variables[i] = []

        self.initTrainingPlot()

    def initTrainingPlot(self):
        if DISPLAY:
            plt.figure(1, figsize = (20,10))
            plt.clf()
            #plt.show()
        
            self.lossPanel = plt.subplot2grid((4, 1), (0, 0), rowspan=2)
    
            self.lossAxes = plt.subplot(4, 1, 3)
            self.variableLines = []
            handles = []
            for i in range(0, self.numLabels):
                thisLine, = self.lossAxes.plot(self.epoch, self.variables[i], linewidth=2, color=LINE_COLORS[i], label=self.labels[i])
                self.variableLines.append(thisLine)
                handles.append(thisLine)
    
            self.lossAxes.legend(handles=handles)
            #self.lossAxes.set_xlabel('Number of epochs')
            self.lossAxes.set_ylabel('Loss')
    
            handles = []
    
            self.variableLocalLines = []
            localX = self.epoch[-1 - self.LOCAL_WINDOW:]
            self.localAxes = plt.subplot(4, 1, 4)
            for i in range(0, self.numLabels):
                localValue = self.variables[i][-1 - self.LOCAL_WINDOW:]
                thisLine, = self.localAxes.plot(localX, localValue, linewidth=2, color=LINE_COLORS[i], label='Local {}'.format(self.labels[i]))
                self.variableLocalLines.append(thisLine)
                handles.append(self.variableLocalLines[i])
                
            self.localAxes.set_ylabel('Loss')
            self.localAxes.set_xlabel('Number of epochs')
    
            
            self.localAxes.legend(handles=handles)

        # scoreAxes = plt.subplot(3, 1, 3)
        # scoreLine, = scoreAxes.plot(scoreX, scoreY, 'k-', label='Score', linewidth=2)

        # scoreAxes.legend(handles=[scoreLine])
        # scoreAxes.set_xlabel('Frames processed')

#        plt.show()

    def updateTrainingData(self, frameNum, variables, shouldSave):
        self.epoch.append(frameNum)
        for i in range(0, self.numLabels):
            self.variables[i].append(variables[i])
            if DISPLAY:
                self.variableLines[i].set_xdata(self.epoch)
                self.variableLines[i].set_ydata(self.variables[i])

        if DISPLAY:
            yMin = 99999
            yMax = -99999
            for i in range(0, self.numLabels):
                yMin = min(yMin, min(self.variables[i]))
                yMax = max(yMax, max(self.variables[i]))
    
            xMax = max(max(self.epoch), self.LOCAL_WINDOW)
    
            self.lossAxes.set_xlim(min(self.epoch) - 0.1, xMax + 0.1)
            self.lossAxes.set_ylim(yMin - 0.001, yMax + 0.001)




        localX = self.epoch[-1 - self.LOCAL_WINDOW:]
        localVariables = []
        for i in range(0, self.numLabels):
            localVariables.append([])
            localVariables[i] = self.variables[i][-1 - self.LOCAL_WINDOW:]
            if DISPLAY:
                self.variableLocalLines[i].set_xdata(localX)
                self.variableLocalLines[i].set_ydata(localVariables[i])

        if DISPLAY:
            if len(localX) > 0:
                yMin = 99999
                yMax = -99999
                for i in range(0, self.numLabels):
                    yMin = min(yMin, min(localVariables[i]))
                    yMax = max(yMax, max(localVariables[i]))
                
                xMax = max(max(localX), min(localX) + self.LOCAL_WINDOW)
    
                self.localAxes.set_xlim(min(localX) - 0.1, xMax + 0.1)
                self.localAxes.set_ylim(yMin - 0.001, yMax + 0.001)

        #yMin = min(scoreY)
        #yMax = max(scoreY)

        #scoreAxes.set_xlim(min(scoreX) - 0.1, max(scoreX) + 0.1)
        #scoreAxes.set_ylim(yMin - 1, yMax + 1)

        if shouldSave:
            dictionary = {}
            dictionary['x'] = self.epoch
            #dictionary['loss'] = self.lossY
            #dictionary['score'] = scoreY
            for i in range(0, self.numLabels):
                dictionary[self.labels[i]] = self.variables[i]

            scipy.io.savemat(self.saveFile, mdict=dictionary)

        if DISPLAY:
            plt.show()
            plt.pause(1e-17)

#modelDirectory = sys.argv[1]


if __name__ == "__main__":
    try:
        indices = np.arange(1,101)
    #    indices = np.arange(4,101)
        experimentNames = ['{}/lstm/{}/'.format(EXP_NAME, i) for i in indices]    
                    
        copyDir = ''
        
        MAX_EPOCHS = 20000
        for i in range(0, len(experimentNames)):
            experimentName = experimentNames[i]
            rootDir = '{}/../'.format(os.path.dirname(os.path.abspath(__file__)))
    #        rootDir = 'F:\Dropbox\ConservationOfAgentDynamics\WorkingMemoryTask/'
            experimentDir = '{}Experiments/{}/'.format(rootDir,experimentName)
            
            if copyDir == '':
                copyDir = '{}/'.format(experimentDir)
                isOriginalRun = True
            
            experimentScript = '{}learnTask.py'.format(experimentDir)
            experimentDataScript = '{}buildData.py'.format(experimentDir)
            experimentSimulation = '{}simulateTask.py'.format(experimentDir)
            
    #        experimentParameters = '{}parameters.pkl'.format(experimentDir)
            
            
            
            print('  Loading data...')
            
            inputData, targetData, classes = buildData(1)
            
            #trainingParams = loadMatlabData('{}pythonParameters.mat'.format(modelDirectory))
            #trainingData = loadMatlabData(trainingParams['dataFile'].format(modelDirectory))
            
            if not os.path.isdir(experimentDir):
                os.makedirs(experimentDir)
            
            # Build data sequences
            print('  Initializing sequences...')
            
            print('  Building sequences...')
            
            data = WeightedSequences()
            if len(inputData.shape) == 3: # Trial data
                data.initTrials(inputData, targetData, classes)
                inputSize = inputData.shape[0]
            else: # Continous data
                data.initContinous(inputData, targetData, classes, NUM_SEQUENCES, SEQUENCE_LENGTH)
                inputSize = inputData.shape[1]
            
            # Train
            
            print('  Starting training...')
            
            model = NNNModel(inputSize, int(np.max(targetData)+1), NUM_NEURONS, MODULE_COUNT, MODULE_CONNECTEDNESS)
            startEpoch = model.tryLoad('{}/savedModels/'.format(experimentDir))
            model.double()
            
            optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4, weight_decay=WEIGHT_DECAY)
            trainloader=DataLoader(dataset=data,batch_size=256, shuffle=True)
            
            
                
            if not '/Experiments/' in os.getcwd():
                if os.path.exists(experimentScript):
                    shutil.copyfile(experimentScript, '{}_backup'.format(experimentScript))
                    shutil.copyfile(experimentDataScript, '{}_backup'.format(experimentDataScript))
                    shutil.copyfile(experimentSimulation, '{}_backup'.format(experimentSimulation))
                    
                if isOriginalRun:                
                    shutil.copyfile(__file__, experimentScript)
                    shutil.copyfile('{}Source/buildData.py'.format(rootDir), experimentDataScript)
                    shutil.copyfile('{}Source/simulateTask.py'.format(rootDir), experimentSimulation)
                    isOriginalRun = False
                else:
                    shutil.copyfile('{}learnTask.py'.format(copyDir), experimentScript)
                    shutil.copyfile('{}buildData.py'.format(copyDir), experimentDataScript)
                    shutil.copyfile('{}simulateTask.py'.format(copyDir), experimentSimulation)
            
            trainingPlot = TrainingPlot('{}trainingData.mat'.format(experimentDir), startEpoch, LABELS)
            
            for epoch in range(startEpoch, MAX_EPOCHS):
                epochLoss = []
                epochAccuracy = []
            
                startTime = time.time()
            
                for inputs, targets, classIDs in trainloader:
                    predicted, dynamics = model(inputs.double(), train=True)
            
                    if DISPLAY:
                        batchLoss, accuracy = loss(predicted, targets, classIDs, trainingPlot.lossPanel)
                    else:
                        batchLoss, accuracy = loss(predicted, targets, classIDs, None)
            
                    optimizer.zero_grad()
            
                    batchLoss.backward()
                    model.setGrad()
                    optimizer.step()
                    epochLoss.append(batchLoss)
                    epochAccuracy.append(accuracy)
            
                losses = [epochLoss[k].cpu().detach().numpy() for k in range(len(epochLoss))]
                meanLoss = np.mean(losses)
                
                accuracy = [epochAccuracy[k].cpu().detach().numpy() for k in range(len(epochAccuracy))]
                meanAccuracy = np.mean(accuracy)
                
                trainingPlot.updateTrainingData(epoch, [meanLoss, meanAccuracy], epoch%10 == 0)
                
                model.saveBackup(epoch)
                
                targetValue = MIN_EPOCH**EPOCH_POWER * MIN_ACCURACY**ACCURACY_POWER / MAX_LOSS**LOSS_POWER
                currentValue = epoch**EPOCH_POWER * meanAccuracy**ACCURACY_POWER / meanLoss**LOSS_POWER
            
                print('Epoch time: {}, accuracy: {}, loss: {}, target: {} > {}'.format(time.time() - startTime, meanAccuracy, meanLoss, currentValue, targetValue))
                
                if currentValue >= targetValue:
                    break;
    finally:
        if DISPLAY:
            plt.close()
    #        plt.draw()
        test = 1
        
