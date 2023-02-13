from lib.plot import plot
from lib.uninterruptibleSimulation import uninterruptibleSimulation
from time import localtime , time,asctime

def main():
    simulation = uninterruptibleSimulation.Test()
    simulation.uninterruptible()
    output = plot.Plot(simulation.testResult)
    output.power()
    output.plotUninterruptible()
    output.price()
    output.plotReward()
    output.fig.tight_layout()    
    output.fig.savefig('lib/plot/Unint/'+str(asctime(localtime(time())))+'.png')

if __name__ == '__main__':
    main()