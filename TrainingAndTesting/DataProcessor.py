import librosa
import os
import numpy as np
import pandas as pd
import time

from glob import glob

# DataPreProcessor is used to extract and process data from our data folder for training

class DataPreprocessor:
    def __init__(self, audioFileName, debug):
        self.inputData = list()
        self.expectedOutputs = list()
        self.audioFileName = audioFileName

        # Saves all the top level directories in /data/
        self.F1 = glob('data/*/')

        # Used to track which folder in /data/ we're on
        self.F1StepCount = 0

        # Key of covid statuses for turning strings from csv into integers to plug into neural network
        self.covidStatusKey = self.fillCovidStatusKey()

        self.debug = debug

        self.maxDuration = 0

    def getInputData(self):
        return self.inputData
    def getExpectedOutputs(self):
        return self.expectedOutputs

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
        csvPath = currentF1Path + os.path.basename(os.path.normpath(currentF1Path)) + ".csv"
        csvFile = pd.read_csv(csvPath)

        if self.debug:
            start = time.time()
            print("Step: " + str(self.F1StepCount) + "/" + str(len(self.F1)))
            print("Current Path: " + currentF1Path)
            # print("CSV Path: " + csvPath)
            # print("F2Dirs: \n\n")

        self.F1StepCount += 1

        # Gets MFCCS values for each audio file, stores in inputData, stores expected value in expectedOutputs
        for dirPath in F2Dirs:
            filePath = dirPath + self.audioFileName

            # if self.debug:
                # print("filePath: " + filePath)

            # Gets and stores mfccs of audio file
            # For any missing audio files/damaged audio files, skip
            try:
                audio, sampleRate = librosa.load(filePath)

                if self.debug:
                    if self.maxDuration < librosa.get_duration(y=audio, sr=sampleRate):
                        self.maxDuration = librosa.get_duration(y=audio, sr=sampleRate)
                        print(self.maxDuration)

                mfccs = librosa.feature.mfcc(y=audio, sr=sampleRate, n_mfcc=40)
                self.inputData.append(mfccs)

                # Gets last directory in the path dirPath (this gives us the individual's identifier)
                id = os.path.basename(os.path.normpath(dirPath))

                # Turns covidStatus val (a string) from dataframe into an ordered set so that {1,0} = pos, {0, 1} = neg, appends tou expectedOutputs
                # covidStatus = csvFile._get_value(id, 'covid_status')
                covidStatus = csvFile.loc[csvFile['id'] == id]['covid_status'].iloc[0]
                covidStatus = self.covidStatusStringToStatusArray(covidStatus)
                self.expectedOutputs.append(covidStatus)
            except:
                if self.debug:
                    print(filePath + " was passed")

        if self.debug:
            print("Time for Step: " + str(time.time() - start) + "seconds")
        return True

    # Given a string covidStatus from 'covid_status" in our csvs, determine if the person is pos/neg and return
    # {x,y} = {1,0} = pos or {x,y} = {0,1} = neg. If encountered with a value never seen before, throw an error.
    def covidStatusStringToStatusArray(self, covidStatus):
        return self.covidStatusKey[covidStatus]

    def fillCovidStatusKey(self):
        key = {}
        # TODO: consider moving positive_asymp to a sign of covid negative -> focus on training ONLY for distinguishing between covid and non covid resp illnesses
        positive = {'positive_mild', 'positive_asymp', 'positive_moderate'}
        negative = {'healthy', 'resp_illness_not_identified', 'no_resp_illness_exposed', 'recovered_full'}
        for item in positive:
            key[item] = np.array([1, 0])
        for item in negative:
            key[item] = np.array([0, 1])
        return key



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
