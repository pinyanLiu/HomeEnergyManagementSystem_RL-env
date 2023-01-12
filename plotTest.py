from lib.plot import plot
from lib.uninterruptibleSimulation import uninterruptibleSimulation


def main():
    simulation = uninterruptibleSimulation.Test()
    simulation.uninterruptible()
    output = plot.Plot(simulation.testResult)
    output.power()
    output.plotUninterruptible()
    output.price()
    output.fig.tight_layout()
    output.fig.savefig('lib/plot/testing.png')

if __name__ == '__main__':
    main()