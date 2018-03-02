class Crop:

    def __init__(self, data, sessionType, index):
        self.data = data
        self.sessionType = sessionType
        self.index = index

    def getData(self):
        return self.data

    def getSessionType(self):
        return self.sessionType

    def getIndex(self):
        return self.index