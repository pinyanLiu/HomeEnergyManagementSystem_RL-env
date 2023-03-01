from lib.Simulations.uninterruptibleSimulation import UnIntSimulation
from lib.Simulations.interruptibleSimulation import IntSimulation

def __main__():
    simulation = IntSimulation()
    simulation.simulation()
    simulation.outputResult()



if __name__ == '__main__':
    __main__()