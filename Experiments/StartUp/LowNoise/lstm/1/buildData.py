# -*- coding: utf-8 -*-
#%%
import numpy as np
    
NUM_SEQUENCES = 300
INPUT_LENGTH = 1000

#%%

def isInBounds(x, y):
    BOX_SIZE = 20
    
    if x < -BOX_SIZE or y < -BOX_SIZE or x > BOX_SIZE or y > BOX_SIZE:
        return False
    
    return True

def buildData(isTraining):
    
    ANGLE_SIGMA = 0.4
    SPEED_LAMBDA = 2
    PAUSE_RATE = 9/10
    PAUSE_TIME = 1
    
    inputData = np.zeros([NUM_SEQUENCES, INPUT_LENGTH, 2])
    targetData = np.zeros([NUM_SEQUENCES, INPUT_LENGTH, 2])
    
    for j in range(0, NUM_SEQUENCES):
        X = np.zeros([INPUT_LENGTH,2])
        Y = np.zeros([INPUT_LENGTH,2])
        
        X[0,:] = 0
    
        nextUnpauseTime = 0;
        
        for i in range(1, X.shape[0]):
            currentSigma = ANGLE_SIGMA
            currentSpeedLambda = SPEED_LAMBDA;
            
            if i >= nextUnpauseTime:
                if np.random.uniform() < PAUSE_RATE:
                    nextUnpauseTime = i + PAUSE_TIME
                    newSpeed = 0
                else:
                    newSpeed = np.random.exponential(currentSpeedLambda)
            else:
                newSpeed = 0
                    
            while True:
                newAngle = X[i-1,0] + np.random.normal(0, currentSigma, 1)    
                velocity = np.array([np.cos(newAngle) * newSpeed, np.sin(newAngle) * newSpeed])
                newPosition = np.add(Y[i-1,:], np.transpose(velocity))
                
                currentSigma = currentSigma * 2
                
                if isInBounds(newPosition[0,0], newPosition[0,1]):
                    break
            
            newAngle = (newAngle + np.pi) % (2 * np.pi) - np.pi
            X[i,0] = newAngle
            X[i,1] = newSpeed
            Y[i,:] = newPosition

        thisTargets = np.insert(Y[0:-1,:], 0, np.array((0, 0)), 0)   
        thisTargets = thisTargets.reshape(thisTargets.shape[0],thisTargets.shape[1])
             
        inputData[j,:,:] = X
        targetData[j,:,:] = thisTargets
        
        if j % (NUM_SEQUENCES/10) == 0:
            print("\tFinished {} of {} traces.".format(j, NUM_SEQUENCES))
    
    classData = np.zeros(NUM_SEQUENCES)
            
    #%%
    
    inputData = np.transpose(inputData, (2, 1, 0))
    targetData = np.transpose(targetData, (2, 1, 0))
    
    return inputData, targetData, classData

if __name__ == "__main__":
    inputData, targetData, classData = buildData(1)
    