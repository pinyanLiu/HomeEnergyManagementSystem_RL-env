import pymysql as pm
import pandas as pd 

class ImportData:
    def __init__(self,host,user,passwd,db,port=3306,charset='utf8'):
        self.host = host
        self.port = port
        self.user = user
        self.passwd = passwd
        self.db = db
        self.charset = charset
        self.__mysqlInit__()

    def __del__(self):
        self.cursor.close()
        self.conn.close()
        print('disconnect mysql')
        
    def __mysqlInit__(self):
        try:
            self.conn = pm.connect(host=self.host,port=self.port,user=self.user,passwd=self.passwd,db=self.db,charset=self.charset)
            self.cursor = self.conn.cursor()
            print(" connect to mysql successfully")
        except:
            print("fail to connect to mysql")


    def importBaseParameter(self):
        self.cursor.execute("SELECT * FROM BaseParameter")
        BaseParameter = pd.DataFrame(self.cursor.fetchall())
        BaseParameter.columns = [column[0] for column in self.cursor.description]
        return BaseParameter    

    def importGridPrice(self):
        self.cursor.execute("SELECT * FROM grid_price")
        gridPrice = pd.DataFrame(self.cursor.fetchall())
        gridPrice.columns = [column[0] for column in self.cursor.description]
        return gridPrice

    def importPhotoVoltaic(self):
        self.cursor.execute("SELECT * FROM PhotoVoltaic")
        PhotoVoltaic = pd.DataFrame(self.cursor.fetchall())
        PhotoVoltaic.columns = [column[0] for column in self.cursor.description]
        return PhotoVoltaic

    def importDeltaSOC(self):
        self.cursor.execute("SELECT * FROM DeltaSOC")
        DeltaSOC = pd.DataFrame(self.cursor.fetchall())
        DeltaSOC.columns = [column[0] for column in self.cursor.description]
        return DeltaSOC

    def importTrainingLoad(self):
        self.cursor.execute("SELECT * FROM TrainingData")
        TrainingData = pd.DataFrame(self.cursor.fetchall())
        TrainingData.columns = [column[0] for column in self.cursor.description]
        return TrainingData

    def importTestingLoad(self):
        self.cursor.execute("SELECT * FROM TestingData")
        TestingData = pd.DataFrame(self.cursor.fetchall())
        TestingData.columns = [column[0] for column in self.cursor.description]
        return TestingData

    def importTemperatureC(self):
        self.cursor.execute("SELECT * FROM TemperatureC")
        TemperatureC = pd.DataFrame(self.cursor.fetchall())
        TemperatureC.columns = [column[0]for column in self.cursor.description]
        return TemperatureC

    def importTemperatureF(self):
        self.cursor.execute("SELECT * FROM TemperatureF")
        TemperatureF = pd.DataFrame(self.cursor.fetchall())
        TemperatureF.columns = [column[0]for column in self.cursor.description]
        return TemperatureF

    def importUserSetTemperatureF(self):
        self.cursor.execute("SELECT * FROM userSetTemperatureF")
        UserSetTemperatureF = pd.DataFrame(self.cursor.fetchall())
        UserSetTemperatureF.columns = [column[0]for column in self.cursor.description]
        return UserSetTemperatureF
    
    def importUserSetTemperatureF2(self):
        self.cursor.execute("SELECT * FROM userSetTemperatureF2")
        UserSetTemperatureF2 = pd.DataFrame(self.cursor.fetchall())
        UserSetTemperatureF2.columns = [column[0]for column in self.cursor.description]
        return UserSetTemperatureF2
    
    def importUserSetTemperatureF3(self):
        self.cursor.execute("SELECT * FROM userSetTemperatureF3")
        UserSetTemperatureF3 = pd.DataFrame(self.cursor.fetchall())
        UserSetTemperatureF3.columns = [column[0]for column in self.cursor.description]
        return UserSetTemperatureF3

    def importUserSetTemperatureC(self):
        self.cursor.execute("SELECT * FROM userSetTemperatureC")
        UserSetTemperatureC = pd.DataFrame(self.cursor.fetchall())
        UserSetTemperatureC.columns = [column[0]for column in self.cursor.description]
        return UserSetTemperatureC

    def importStatisticalData(self):
        self.cursor.execute("SELECT * FROM TestResult_Statistical_Data")
        statisticalData = pd.DataFrame(self.cursor.fetchall())
        statisticalData.columns = [column[0]for column in self.cursor.description]
        return statisticalData
    
    def importOccupancy(self):
        self.cursor.execute("SELECT * FROM occupancy")
        Occupancy = pd.DataFrame(self.cursor.fetchall())
        Occupancy.columns = [column[0]for column in self.cursor.description]
        return Occupancy
    
    def importOccupancyTest(self):
        self.cursor.execute("SELECT * FROM occupancyTest")
        OccupancyTest = pd.DataFrame(self.cursor.fetchall())
        OccupancyTest.columns = [column[0]for column in self.cursor.description]
        return OccupancyTest

    def importIntPreference(self,id):
        self.cursor.execute("SELECT * FROM intPreference"+str(id))
        userPreference = pd.DataFrame(self.cursor.fetchall())
        userPreference.columns = [column[0]for column in self.cursor.description]
        return userPreference
    
    def importUnIntPreference(self,id):
        self.cursor.execute("SELECT * FROM unIntPreference"+str(id))
        unIntPreference = pd.DataFrame(self.cursor.fetchall())
        unIntPreference.columns = [column[0]for column in self.cursor.description]
        return unIntPreference
