import pymysql as pm
import pandas as pd 

class ImportData:
    def __init__(self,host,user,passwd,db,mode,port=3306,charset='utf8'):
        self.host = host
        self.port = port
        self.user = user
        self.passwd = passwd
        self.db = db
        self.charset = charset
        self.__mysqlInit__()
        self.BaseParameter = self.__importBaseParameter__()
        self.GridPrice = self.__importGridPrice__()
        self.PV = self.__importPhotoVoltaic__()
        if mode == 'Training':
            self.Load = self.__importTrainingLoad__()
        elif mode == 'Testing':
            self.Load = self.__importTestingLoad__()
        self.experimentData = {'BaseParameter':self.BaseParameter,'GridPrice':self.GridPrice,'PV':self.PV,'Load':self.Load}

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
            print("fail to connet to mysql")


    def __importBaseParameter__(self):
        self.cursor.execute("SELECT * FROM BaseParameter")
        BaseParameter = pd.DataFrame(self.cursor.fetchall())
        BaseParameter.columns = [column[0] for column in self.cursor.description]
        return BaseParameter    

    def __importGridPrice__(self):
        self.cursor.execute("SELECT * FROM grid_price")
        gridPrice = pd.DataFrame(self.cursor.fetchall())
        gridPrice.columns = [column[0] for column in self.cursor.description]
        return gridPrice

    def __importPhotoVoltaic__(self):
        self.cursor.execute("SELECT * FROM PhotoVoltaic")
        PhotoVoltaic = pd.DataFrame(self.cursor.fetchall())
        PhotoVoltaic.columns = [column[0] for column in self.cursor.description]
        return PhotoVoltaic

    def __importTrainingLoad__(self):
        self.cursor.execute("SELECT * FROM TrainingData")
        TrainingData = pd.DataFrame(self.cursor.fetchall())
        TrainingData.columns = [column[0] for column in self.cursor.description]
        return TrainingData

    def __importTestingLoad__(self):
        self.cursor.execute("SELECT * FROM TestingData")
        TestingData = pd.DataFrame(self.cursor.fetchall())
        TestingData.columns = [column[0] for column in self.cursor.description]
        return TestingData

    def __importTemperature__(self):
        self.cursor.execute("SELECT * FROM Temperature")
        Temperature = pd.DataFrame(self.cursor.fetchall())
        Temperature.columns = [column[0]for column in self.cursor.description]
        return Temperature