from lib.Simulations.multiSimulation import multiSimulation
import pandas as pd
from sqlalchemy import create_engine

simulation = multiSimulation()
simulation.simulation()
testResult = simulation.hrlTestResult
# for i in range(12):
#     print(testResult[i]['deltaSoc'])

deltaSOC = pd.DataFrame([],columns=['timeBlock','layer 1','layer 2','layer 3','layer 4','layer 5','layer 6','layer 7','layer 8'])
for i in range(96):
    deltaSOC.loc[i,'timeBlock'] = i

for months in range(12):
    if months == 0:
        deltaSOC['Jan'] = testResult[months]['deltaSoc']
    elif months == 1:
        deltaSOC['Feb'] = testResult[months]['deltaSoc']
    elif months == 2:
        deltaSOC['Mar'] = testResult[months]['deltaSoc']
    elif months == 3:
        deltaSOC['Apr'] = testResult[months]['deltaSoc']
    elif months == 4:
        deltaSOC['May'] = testResult[months]['deltaSoc']
    elif months == 5:
        deltaSOC['Jun'] = testResult[months]['deltaSoc']
    elif months == 6:
        deltaSOC['July'] = testResult[months]['deltaSoc']
    elif months == 7:
        deltaSOC['Aug'] = testResult[months]['deltaSoc']
    elif months == 8:
        deltaSOC['Sep'] = testResult[months]['deltaSoc']
    elif months == 9:
        deltaSOC['Oct'] = testResult[months]['deltaSoc']
    elif months == 10:
        deltaSOC['Nov'] = testResult[months]['deltaSoc']
    elif months == 11:
        deltaSOC['Dcb'] = testResult[months]['deltaSoc']

print(deltaSOC)
tableName = "DeltaSOC"
dataframe = pd.DataFrame(data=deltaSOC)


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

