from lib.Simulations.uninterruptibleSimulation import UnIntSimulation
from lib.Simulations.interruptibleSimulation import IntSimulation
from lib.Simulations.SocSimulation import SocSimulation
from lib.Simulations.hvacSimulation import HvacSimulation
from lib.Simulations.multiSimulation import multiSimulation
from lib.Simulations.FourLevelSimulation import FourLevelSimulation
from lib.Simulations.OneLevelSimulation import OneLevelSimulation
import sys

def __main__(argv):
    if len(argv)<2:
        print('please give parameters Testing mode: "soc"/"int"/"hvac"/"unint"/"hrl"/"4"')
        return
    mode = argv[1]
    if mode == 'unint':
        simulation = UnIntSimulation()
    elif mode == 'int':
        simulation = IntSimulation()
    elif mode == "soc":
        simulation = SocSimulation()
    elif mode == "hvac":
        simulation = HvacSimulation()
    elif mode == "hrl":
        simulation = multiSimulation()
    elif mode == "4":
        simulation = FourLevelSimulation()
    elif mode == "1":
        simulation = OneLevelSimulation()
    else:
        print('please give parameters Testing mode: "soc"/"int"/"hvac"/"unint"/"hrl"')
        return 
    simulation.hrlSimulation()
    if mode == "hrl" or mode == "4" or mode == "1":
        # simulation.EachMonthResult()
        # simulation.avgMonthResult()
        #simulation.hrlSimulation()
        print("heeheee")
        # simulation.outputResult("soc")
        # simulation.outputResult("hvac")
        # simulation.outputResult("int")
        # simulation.outputResult("unint")
        # simulation.outputResult("")
        # simulation.plotessResult(5)
        # simulation.plotacResult(1,5)
        # simulation.plotacResult(2,5)
        # simulation.plotacResult(3,5)
        # simulation.plotintResult(1,5)
        # simulation.plotintResult(2,5)
        # simulation.plotintResult(3,5)
        # simulation.plotunintResult(1,5)
        # simulation.plotunintResult(2,5)
    else:
        for month in range(1,12):
            simulation.outputResult(id=1,month=month)
    



if __name__ == '__main__':
    __main__(sys.argv)