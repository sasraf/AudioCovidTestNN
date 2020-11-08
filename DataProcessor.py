class DataPreprocessor:
    def __init__(self, audioFileName):
        self.inputData = []
        self.expectedOutputs = []
        self.fileName = audioFileName

    # NOTE: audioFileName is the name of the file we're looking at. So if we're looking at "breathing_shallow.wav" then filename = "breathing_shallow.wav"

    # TODO: save one folder's worth of data to inputData, expectedOutputs
    # Gets and saves all the data from one data folder titled "2020..."
    def getOneSetOfData(self, fileName):
        # NOTE: fileName is the name of the data folder. Ex: "20200413"

        # Clear both lists
        self.inputData.clear()
        self.expectedOutputs.clear()

        return self.inputData, self.expectedOutputs




    # TODO: save one person's worth of data to inputData, expectedOutputs
    # Get the data from one person
    def getOnePersonsData(self, fileName):
        return None

