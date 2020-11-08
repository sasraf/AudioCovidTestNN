import pandas as pd
import os

class DataPreprocessor:
    def __init__(self, audioFileName):
        self.inputData = list()
        self.expectedOutputs = list()
        self.fileName = audioFileName

    # NOTE: audioFileName is the name of the file we're looking at. So if we're looking at "breathing_shallow.wav" then filename = "breathing_shallow.wav"

    # TODO: save one folder's worth of data to inputData, expectedOutputs
    # Gets and saves all the data from one data folder titled "2020..."
    def getOneSetOfData(self, dateName):
        # NOTE: dateName is the name of the data folder. Ex: "20200413"

        # Clear both lists
        self.inputData.clear()
        self.expectedOutputs.clear()

        # Save csv as a dataframe
        df = pd.read_csv('/data/' + dateName + '/' + dateName + '.csv')

        # Get a list of all directories in this folder (the list of all the individual people's files)
        dirs = list()
        dirs = next(os.walk('/data/' + dateName + '/'))[1]

        # TODO traverse peoplefolders, for each person's folder add the mfcc value to inputData and get expected output value via pandas



        return self.inputData, self.expectedOutputs




    # # TODO: save one person's worth of data to inputData, expectedOutputs
    # # Get the data from one person
    # def getOnePersonsData(self, dateName, fileName):
    #     # NOTE: fileName is the
    #     df = data
    #     return None

