import imp
import pymysql as pm
import pandas as pd 
from random import choice
class ImportData:
    def __init__(self,host,user,passwd,db,port=3306,charset='utf8'):
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
        self.Load = self.__importLoad__()
        self.experimentData = {'BaseParameter':self.BaseParameter,'GridPrice':self.GridPrice,'PV':self.PV,'Load':self.Load}

    def __del__(self):
        self.cursor.close()
        self.conn.close()
        print('disconnect mysql')
        
    def __mysqlInit__(self):
        self.conn = pm.connect(host=self.host,port=self.port,user=self.user,passwd=self.passwd,db=self.db,charset=self.charset)
        self.cursor = self.conn.cursor()

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

    def __importLoad__(self):
        self.cursor.execute("SELECT * FROM sum_of_total_load_per15min")
        sum_of_total_load_per15min = pd.DataFrame(self.cursor.fetchall())
        sum_of_total_load_per15min.columns = [column[0] for column in self.cursor.description]
        return sum_of_total_load_per15min

if __name__=='__main__':
    EPData = ImportData(host="140.124.42.65",user= "root",passwd= "fuzzy314",db= "Cems_data")
#    print(EPData.experimentData['BaseParameter'].loc[EPData.experimentData['BaseParameter']['parameter_name']=='batteryCapacity',['value']])
    # print(EPData.experimentData['GridPrice'])
    # print(EPData.experimentData['PV'])
    # print(EPData.experimentData['Load'])
    print(EPData.experimentData['Load'].iloc[:,1])