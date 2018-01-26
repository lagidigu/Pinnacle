import os
import pickle
import random

import numpy as np
from Meditation.Timemarker_Classification.RawConverter import RawConverter
from Meditation.Timemarker_Classification.SQLImporter import RecordingType
from Meditation.Timemarker_Classification.SQLImporter import SQLImporter
from Meditation.Timemarker_Classification.SessionCropper import SessionCropper


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
        self.numOfTimeMarkerSessions = 4
        self.dayDreamSessionName = 'DayDream_'
        self.mentalMathSessionName = 'MentalMath_'
        self.youtubeSessionsName = 'YouTube_'
        self.meditationSessionName = 'MeditationSession_'
        self.eyesClosedSessionName = 'ControlSession_'
        self.timeMarkerSessionName = 'TimeMarkerSession_'

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
        for i in range(1, self.numOfTimeMarkerSessions + 1):
            rawConverter = self.createRawConverter(self.timeMarkerSessionName, i)
            rawSessionList.append(rawConverter)
        return rawSessionList



    def getCrops(self, rawSessionList, useCropping = True): #TODO: Check if the time marker is set to start or end of focus interval
        if useCropping:
            crops = SessionCropper(rawSessionList, 500, 1).crops
        else:
            crops = SessionCropper(rawSessionList, 500, 500).crops
        focus = 0
        normal = 0
        for i in range (0, len(crops)):
            if crops[i].sessionType == RecordingType.Focus:
                focus += 1
            if crops[i].sessionType == RecordingType.Meditation:
                normal += 1
        self.overSample(crops, int(normal/focus)) #This is only for the time marker class
        mySeed = 666
        random.seed(mySeed)
        random.shuffle(crops)
        return crops

    def overSample(self, crops, factor):
        print("Oversampling.")
        for i in range (0, len(crops)):
            if (crops[i].sessionType == RecordingType.Focus):
                for j in range(0, factor):
                    crops.append(crops[i])


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
        else:
            sessions = self.createRawSessionList()
            self.saveToPickle(sessions, 'Sessions')
        return sessions

    # Method to be called from the Neural Network
    def getFeaturesAndLabels(self, useCropping = True):
        sessions = self.getSessions(useLoadedData=False)
        print("Sessions Acquired.")
        crops = self.getCrops(sessions, useCropping)
        print("Crops Acquired and Shuffled.")
        trainX, trainY, testX, testY, validationX, validationY = self.createFeaturesAndLabels(crops)
        print("Features Created.")
        return trainX, trainY, testX, testY, validationX, validationY

    def TEST_distribution_ratio(self, crops):
        meditation = 0
        eyes_closed = 0

        for i in range (0, len(crops)):
            if i % 10 == 0:
                print(crops[i].sessionType)
            # if crops[i].sessionType == RecordingType.Meditation:
            #     meditation += 1
            # if crops[i].sessionType == RecordingType.EyesClosed:
            #     eyes_closed += 1
        # print(len(crops), meditation, eyes_closed, meditation + eyes_closed)

#TODO: Use stratified, instead of random sampling


