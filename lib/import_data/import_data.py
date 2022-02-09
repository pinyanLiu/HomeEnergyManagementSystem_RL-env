import pymysql
import pandas as pd 

class import_data:
    def __init__(self,host,port,user,passwd,db,charset='utf8'):
        self.host = host
        self.port = port
        self.user = user
        self.passwd = passwd
        self.db = db
        self.charset = charset
        self.BaseParameter = pd.DataFrame()
        self.GridPrice = pd.DataFrame()
        self.PV = pd.DataFrame()
        self.Load = pd.DataFrame()
        self.experimentData = {'BaseParameter':self.BaseParameter,'GridPrice':self.GridPrice,'PV':self.PV,'Load':self.Load}
    def __mysqlInit__(self):
        self.conn = pymysql.connect(host=self.host,port=self.port,user=self.user,passwd=self.passwd,charset=self.charset)
        self.cursor = self.conn.cursor()

    def __importBaseParameter__(self):
        pass
    
    def __importGridPrice__(self):
        pass
    def __importPhotoVoltaic__(self):
        pass
    def __importLoad__(self):
        pass
