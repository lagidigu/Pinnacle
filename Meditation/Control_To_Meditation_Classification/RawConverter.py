import numpy as np
import random

#Shape [maxTimeStep][maxChannels] with the session type is being fed in.
#This class converts the input to a structure with
#2D Array: [activeChannels][currentMaxTimeStep]
#Enum: type of recording session

class RawConverter:

    def __init__(self, rows, sessionType):
        self.rawData = self.processInput(rows)
        self.timeMarkerIndices = self.getTimeMarkerIndices(rows)
        self.sessionType = sessionType

    def processInput(self, rows):
        processedInput = self.flipArray(rows)
        processedInput = self.shortenChannels(processedInput)
        processedInput = self.shortenTimeStep(processedInput)
        return processedInput

    def flipArray(self, rows):
        return np.transpose(rows)

    def getTimeMarkerIndices(self, rows):
        rows = np.transpose(rows)
        timeMarkers = list()
        for i in range (0, 600):  # There will never be more than 600 timeMarkers per Session
            if (rows[8][i] != 0):
                timeMarkers.append(rows[8][i])
        return timeMarkers

    def shortenChannels(self, inputTable):
        return inputTable[0 : 6, 0 : len(inputTable[0])]

    def shortenTimeStep(self, inputTable):
        maxValue = 0
        accumulator = 0
        for i in range(0, len(inputTable)):
            for j in range (0, len(inputTable[0])):
                if (inputTable[i][j] == 0):
                    accumulator = accumulator + 1
                    if (accumulator == 5):
                        localMax = (j + 1) - accumulator
                        if (localMax > maxValue):
                            maxValue = localMax
                else:
                    accumulator = 0
        return inputTable[0 : len(inputTable), 0 : maxValue]
