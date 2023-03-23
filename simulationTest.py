from lib.Simulations.uninterruptibleSimulation import UnIntSimulation
from lib.Simulations.interruptibleSimulation import IntSimulation
from lib.Simulations.SocSimulation import SocSimulation
from lib.Simulations.hvacSimulation import HvacSimulation
from lib.Simulations.multiSimulation import multiSimulation
import sys

def __main__(argv):
    if len(argv)<2:
        print('please give parameters Testing mode: "soc"/"int"/"hvac"/"unint"/"hrl"')
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
    else:
        print('please give parameters Testing mode: "soc"/"int"/"hvac"/"unint"/"hrl"')
        return 
    
    simulation.simulation()
    simulation.outputResult()



if __name__ == '__main__':
    __main__(sys.argv)