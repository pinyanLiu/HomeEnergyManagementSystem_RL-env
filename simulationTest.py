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
    simulation.simulation()
    #simulation.EachMonthResult()
    simulation.outputResult()



if __name__ == '__main__':
    __main__(sys.argv)