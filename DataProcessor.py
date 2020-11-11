import pandas as pd
import librosa
import os

from glob import glob

# DataPreProcessor is used to extract and process data from our data folder for training

class DataPreprocessor:
    def __init__(self, audioFileName):
        self.inputData = list()
        self.expectedOutputs = list()
        self.audioFileName = audioFileName

        # Saves all the top level directories in /data/
        self.F1 = glob('data/*/')

        # Used to track which folder in /data/ we're on
        self.F1StepCount = 0

    # Get all of the data from a single top level F1 folder and save it to our inputData and expectedOutputs
    # If data was retrieved, return true. Else if all data is retreived return false
    # TODO in main: Iterate through every F1 folder by doing x = true \n while(x){ x = getF1Data } \n get input vals, get output vals
    def getF1Data(self):

        # If we've iterated through all the top level F! directories
        # return false
        if self.F1StepCount == len(self.F1):
            return False

        # Saves all subdirectories of current directory, iterates F1StepCount
        currentF1Path = self.F1[self.F1StepCount]
        F2Dirs = glob(currentF1Path + "*/")

        # Stores csv file as pd dataframe
        csvFile = pd.read_csv(currentF1Path + os.path.basename(os.path.normpath(currentF1Path)) + ".csv")

        self.F1StepCount += 1

        # Gets MFCCS values for each audio file, stores in inputData, stores expected value in expectedOutputs
        for dirPath in F2Dirs:
            filePath = dirPath + self.audioFileName

            # Gets and stores mfccs of audio file
            audio, sampleRate = librosa.load(filePath)
            mfccs = librosa.feature.mfcc(y=audio, sr=sampleRate, n_mfcc=40)
            self.inputData.append(mfccs)


            # TODO: process expectedOutputs, append
            # Gets last directory in the path dirPath (this gives us the individual's identifier)
            id = os.path.basename(os.path.normpath(dirPath))

            # Turns covidStatus val (a string) from dataframe into an ordered set so that {1,0} = pos, {0, 1} = neg, appends tou expectedOutputs
            covidStatus = csvFile._get_value(id, 'covid_status')
            covidStatus = self.covidStatusStringToStatusArray(covidStatus)
            self.expectedOutputs.append(covidStatus)

        return True


    # TODO: return values based off of these reqs: If the person is currently positive, {x,y} = {1, 0}. If they're negative {x, y} = {0, 1}
    # Given a string covidStatus from 'covid_status" in our csvs, determine if the person is pos/neg and return
    # {x,y} = {1,0} = pos or {x,y} = {0,1} = neg. If encountered with a value never seen before, throw an error.
    def __covidStatusStringToStatusArray(self, covidStatus):
        return False


    # TODO: get function for inputData, expectedOutput

    # # NOTE: audioFileName is the name of the file we're looking at. So if we're looking at "breathing_shallow.wav" then filename = "breathing_shallow.wav"
    #
    # # TODO: save one folder's worth of data to inputData, expectedOutputs
    # # Gets and saves all the data from one data folder titled "2020..."
    # def getOneSetOfData(self, dateName):
    #     # NOTE: dateName is the name of the data folder. Ex: "20200413"
    #
    #     # Clear both lists
    #     self.inputData.clear()
    #     self.expectedOutputs.clear()
    #
    #     # Save csv as a dataframe
    #     df = pd.read_csv('/data/' + dateName + '/' + dateName + '.csv')
    #
    #     # Get a list of all directories in this folder (the list of all the individual people's files)
    #     dirs = list()
    #     dirs = next(os.walk('/data/' + dateName + '/'))[1]
    #
    #     # TODO traverse peoplefolders, for each person's folder add the mfcc value to inputData and get expected output value via pandas
    #
    #
    #
    #     return self.inputData, self.expectedOutputs
    #
    #
    #
    #
    # # # TODO: save one person's worth of data to inputData, expectedOutputs
    # # # Get the data from one person
    # # def getOnePersonsData(self, dateName, fileName):
    # #     # NOTE: fileName is the
    # #     df = data
    # #     return None
    #
