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

    def importUserSetTemperatureC(self):
        self.cursor.execute("SELECT * FROM userSetTemperatureC")
        UserSetTemperatureC = pd.DataFrame(self.cursor.fetchall())
        UserSetTemperatureC.columns = [column[0]for column in self.cursor.description]
        return UserSetTemperatureC