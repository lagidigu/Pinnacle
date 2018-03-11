from Inference.Control_To_Meditation_Classification.Crop import Crop

class SessionCropper:

    def __init__(self, sessionList : list, size, step):
        self.crops = self.convertSessionCrops(sessionList, size, step)

    def convertSessionCrops(self, sessionList : list, size, step):
        cropList = list()
        for session in sessionList:
            #session : RawConverter
            for i in range (0, len(session.rawData[0]) - size, step):
                tempData = session.rawData[0 : len(session.rawData), i : (i + size)]
                tempCrop = Crop(tempData, session.sessionType, i)
                cropList.append(tempCrop)
        return cropList

    def getSessionCrops(self):
        return self.crops


