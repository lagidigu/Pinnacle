import os
import pickle
import random
import tensorflow as tf

import numpy as np
from Inference.Control_To_Meditation_Classification.RawConverter import RawConverter
from Inference.Control_To_Meditation_Classification.SQLImporter import RecordingType
from Inference.Control_To_Meditation_Classification.SQLImporter import SQLImporter
from Inference.Control_To_Meditation_Classification.SessionCropper import SessionCropper


#This is the Data Structure Parser for Meditation
class dataStructureParser:
    def __init__(self):
        self.sessionList = list()
        self.numOfSessionTypes = 2
        self.numOfDayDreamSessions = 1
        self.numOfMentalMathSessions = 1
        self.numOfYoutubeSessions = 1
        self.numOfMeditationSessions = 6
        self.numOfEyesClosedSessions = 6
        self.dayDreamSessionName = 'DayDream_'
        self.mentalMathSessionName = 'MentalMath_'
        self.youtubeSessionsName = 'YouTube_'
        self.meditationSessionName = 'MeditationSession_'
        self.eyesClosedSessionName = 'ControlSession_'

    def createRawConverter(self, givenName, i):
        name = givenName + str(i)
        tempSQL = SQLImporter(name)
        rawConverter = RawConverter(tempSQL.rawData, tempSQL.sessionType)
        return rawConverter


    def createRawSessionList(self):
        rawSessionList = list()
        for i in range(1, self.numOfMeditationSessions + 1):
            rawConverter = self.createRawConverter(self.meditationSessionName, i)
            rawSessionList.append(rawConverter)
        for i in range(1, self.numOfEyesClosedSessions + 1):
            rawConverter = self.createRawConverter(self.eyesClosedSessionName, i)
            rawSessionList.append(rawConverter)
        return rawSessionList


    def getCrops(self, rawSessionList, useCropping = True):
        if useCropping:
            crops = SessionCropper(rawSessionList, 500, 1).crops
        else:
            crops = SessionCropper(rawSessionList, 500, 500).crops
        mySeed = 666
        random.seed(mySeed)
        random.shuffle(crops)
        return crops

    def saveToPickle(self, ourObj, fileName):
        pickleOut =  open(fileName, 'wb')
        pickle.dump(ourObj, pickleOut)
        pickleOut.close()

    def loadPickle(self, fileName):
        print(os.getcwd())
        pickleIn = open(fileName, 'rb')
        loaded = pickle.load(pickleIn)
        return loaded

    def createFeaturesAndLabels(self, cropList : list, testSize = 0.1, validationSize = 0.05):
        trainSampleSize = int((1 - testSize - validationSize) * len(cropList))
        testSampleSize = int(testSize * len(cropList))
        validationSampleSize = int(validationSize * len(cropList))
        trainList = list(cropList[0 : trainSampleSize])
        testList = list(cropList[trainSampleSize : trainSampleSize + testSampleSize])
        validationList = list(cropList[trainSampleSize + testSampleSize : trainSampleSize + testSampleSize + validationSampleSize])
        random.shuffle(trainList)
        random.shuffle(testList)
        random.shuffle(validationList)
        trainX = []
        trainY = []
        for i in range(0, len(trainList)):
            trainX.append(trainList[i].getData())
            trainY.append(trainList[i].getSessionType().value)
        testX = []
        testY = []
        for i in range(0, len(testList)):
            testX.append(testList[i].getData())
            testY.append(testList[i].getSessionType().value)
        validationX = []
        validationY = []
        for i in range(0, len(validationList)):
            validationX.append(validationList[i].getData())
            validationY.append(validationList[i].getSessionType().value)
        trainY = self.getOneHotEncoding(trainY)
        testY = self.getOneHotEncoding(testY)
        validationY = self.getOneHotEncoding(validationY)
        return trainX, trainY, testX, testY, validationX, validationY

    def getOneHotEncoding(self, myArray):
        targets = np.array(myArray).reshape(-1)
        oneHotEncoded = np.eye(self.numOfSessionTypes)[targets]
        return oneHotEncoded


    def getSessions(self, useLoadedData = True):
        if (useLoadedData):
            sessions = self.loadPickle('Sessions')
            for i in range (0, len(sessions)):
                sessions[i].rawData = np.array(sessions[i].rawData, dtype=np.float32)

            print(type(sessions[0].rawData[0][550]))
        else:
            sessions = self.createRawSessionList()
            print(sessions[0].rawData[0][550])
            self.saveToPickle(sessions, 'Sessions')
        return sessions

    def convertToFloat32(self, trainX, trainY, testX, testY, validationX, validationY):
        newTrainX = np.array(trainX).astype(float)
        newTrainY = np.array(trainY).astype(float)
        newTestX = np.array(testX).astype(float)
        newTestY = np.array(testY).astype(float)
        newValidationX = np.array(validationX).astype(float)
        newValidationY = np.array(validationY).astype(float)
        return newTrainX, newTrainY, newTestX, newTestY, newValidationX, newValidationY



    # Method to be called from the Neural Network
    def getFeaturesAndLabels(self, useCropping = True):
        sessions = self.getSessions(useLoadedData=True)
        print("Sessions Acquired.")
        crops = self.getCrops(sessions, useCropping)
        print("Crops Acquired and Shuffled.")
        trainX, trainY, testX, testY, validationX, validationY = self.createFeaturesAndLabels(crops)
        print("Features Created.")
        #trainX, trainY, testX, testY, validationX, validationY = self.convertToFloat32(trainX, trainY, testX, testY, validationX, validationY)
        #print("Converted to float 32")
        #print(type(trainX[0][0][0]))
        return trainX, trainY, testX, testY, validationX, validationY



