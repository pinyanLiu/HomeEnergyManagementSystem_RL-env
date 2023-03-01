import pandas as pd

class Simulation():
    def __init__(self) -> None:
        self.testResult = {}
        for month in range(12):
            self.testResult[month] = pd.DataFrame()
    
    def simulation(self):
        print("do simulation")
    
    def outputResult(self):
        print("plot result")

    def __del__(self):
        pass
