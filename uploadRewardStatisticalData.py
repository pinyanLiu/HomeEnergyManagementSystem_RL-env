from lib.Simulations.SocSimulation import SocSimulation
from lib.Simulations.hvacSimulation import HvacSimulation
from lib.Simulations.interruptibleSimulation import IntSimulation
from lib.Simulations.uninterruptibleSimulation import UnIntSimulation
import pandas as pd
from sqlalchemy import create_engine

socsimulation = SocSimulation()
socsimulation.simulation()
socmean = socsimulation.getMean()
socstd = socsimulation.getStd()
socMin = socsimulation.getMin()
socMax = socsimulation.getMax()

hvacsimulation = HvacSimulation()
hvacsimulation.simulation()
hvacmean = hvacsimulation.getMean()
hvacstd = hvacsimulation.getStd()
hvacMin = hvacsimulation.getMin()
hvacMax = hvacsimulation.getMax()

Intsimulation = IntSimulation()
Intsimulation.simulation()
Intmean = Intsimulation.getMean()
Intstd = Intsimulation.getStd()
IntMin = Intsimulation.getMin()
IntMax = Intsimulation.getMax()

UnIntsimulation = UnIntSimulation()
UnIntsimulation.simulation()
UnIntmean = UnIntsimulation.getMean()
UnIntstd = UnIntsimulation.getStd()
UnIntMin = UnIntsimulation.getMin()
UnIntMax = UnIntsimulation.getMax()
# for i in range(12):
#     print(testResult[i]['deltaSoc'])

upload = {"name":["SOC","HVAC","Interruptible","Uninterruptible"],
          "mean":[socmean,hvacmean,Intmean,UnIntmean],
          "std":[socstd,hvacstd,Intstd,UnIntstd],
          "Min":[socMin,hvacMin,IntMin,UnIntMin],        
          "Max":[socMax,hvacMax,IntMax,UnIntMax]        
}

tableName = "TestResult_Statistical_Data"
dataframe = pd.DataFrame(data=upload)
print(upload)


engine = create_engine("mysql+pymysql://{}:{}@{}/{}?charset={}".format('root', 'fuzzy314', '140.124.42.65', 'chig_Cems_data','utf8'))
con = engine.connect()

try:

    frame= dataframe.to_sql(tableName, con, if_exists='fail')

except ValueError as vx:

    print(vx)

except Exception as ex:   

    print(ex)

else:
    print("Table %s created successfully."%tableName)
finally:
    con.close()
