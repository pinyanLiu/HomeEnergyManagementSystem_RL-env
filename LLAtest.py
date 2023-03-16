from lib.enviroment.multiAgentEnv.LLA.LLA import socLLA
from  yaml import load , SafeLoader
from  lib.import_data.import_data import ImportData 

if __name__ == '__main__':
    totalState = {
            "sampleTime":0,
            "fixLoad":2,
            "PV":1,
            "SOC":0.7,
            "pricePerHour":3,
            "deltaSoc":0,
            "indoorTemperature":80,
            "outdoorTemperature":80,
            "userSetTemperature":70,
            "intRemain":15,
            "unintRemain":15,
            "unintSwitch":0,
            "order":0
        }    
    with open("yaml/mysqlData.yaml","r") as f:
            mysqlData = load(f,SafeLoader)

    host = mysqlData['host']
    user = mysqlData['user']
    passwd = mysqlData['passwd']
    db = mysqlData['db']
    info = ImportData(host= host ,user= user ,passwd= passwd ,db= db)
    #import Base Parameter
    BaseParameter = info.importBaseParameter()
    soc = socLLA(0.07,0.205,BaseParameter)
    soc.environment.reset()
    soc.getState(totalState)
    soc.environment.updateState(soc.states)
    soc.execute()
    soc.rewardStandardization()
    print(soc.reward)