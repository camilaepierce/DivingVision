import json


class DivingConfig():

    def __init__(self, filename="../config.json"):
        with open(filename) as config_file:
            self.config = json.load()
    
    def getAll():
        return self.config

    def getInfo():
        return self.config.standard

    def trainingConfig():
        return self.config.training


    def evalConfig():
        return self.config.evaluation
