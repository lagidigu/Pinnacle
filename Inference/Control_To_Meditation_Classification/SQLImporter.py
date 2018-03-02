import sqlite3
import numpy as np
from enum import Enum

class SQLImporter:

    def __init__(self, name):
        self.rawData= self.getRawData(name)
        self.sessionType = self.getSessionType(name)

    def getRawData(self, name):
        conn = sqlite3.connect('EEGDataBase.db')
        cur = conn.cursor()
        cur.execute("SELECT * FROM %s" % name)
        rows = cur.fetchall()

        return np.array(rows).astype(float)

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








