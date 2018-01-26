import sqlite3
import numpy as np
from enum import Enum

class SQLImporter:

    def __init__(self, name):
        self.getAllNames()
        self.rawData= self.getRawData(name)
        self.sessionType = self.getSessionType(name)

    def getAllNames(self):
        conn = sqlite3.connect(
            '/informatik2/students/home/5baracat/PycharmProjects/Praktikum-Neuronale-Netze/EEGDataBase.db')
        res = conn.execute("SELECT name FROM sqlite_master WHERE type='table';")
        for name in res:
            print(name[0])

    def getRawData(self, name):
        conn = sqlite3.connect('/informatik2/students/home/5baracat/PycharmProjects/Praktikum-Neuronale-Netze/EEGDataBase.db')
        cur = conn.cursor()
        cur.execute("SELECT * FROM %s" % name)
        rows = cur.fetchall()

        return np.array(rows)

    def getSessionType(self, name):
        if ('Meditation' in name) or ('TimeMarker' in name):
            return RecordingType.Meditation
        if ('Control' in name):
            return RecordingType.EyesClosed
        if ('Day' in name):
            return RecordingType.WindowDreaming
        if ('YouTube' in name):
            return RecordingType.YoutubeWatching
        if ('Mental' in name):
            return RecordingType.MentalMath


class RecordingType(Enum):
    Meditation = 0
    EyesClosed = 1
    WindowDreaming = 2
    YoutubeWatching = 3
    MentalMath = 4








