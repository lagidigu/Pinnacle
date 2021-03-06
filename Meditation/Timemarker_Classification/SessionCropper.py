from Meditation.Timemarker_Classification.Crop import Crop
from Meditation.Timemarker_Classification.SQLImporter import RecordingType

class SessionCropper:

    def __init__(self, sessionList : list, size, step):
        self.crops = self.convertSessionCrops(sessionList, size, step)

    def convertSessionCrops(self, sessionList : list, size, step):
        cropList = list()
        for session in sessionList:
            #session : RawConverter
            for i in range (0, len(session.rawData[0]) - size, step):

                sessionType = session.sessionType
                if (self.checkIfTimerMarkersInCrop(i, session)): #TODO: We assume that index is at beginning of interval
                    sessionType = RecordingType.Focus

                tempData = session.rawData[0 : len(session.rawData), i : (i + size)]
                tempCrop = Crop(tempData, sessionType, i)
                cropList.append(tempCrop)
        return cropList

    def checkIfTimerMarkersInCrop(self, index, rawSession):

        for i in range (0, len(rawSession.timeMarkerIndices)):
            if (rawSession.timeMarkerIndices[i] < index < rawSession.timeMarkerIndices[i] + 2000):
                return True
        return False


    def getSessionCrops(self):
        return self.crops


