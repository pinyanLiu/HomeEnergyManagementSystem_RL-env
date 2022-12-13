from lib.plot import plot
from projects.RL_firstry.lib.test import uninterruptibleSimulation


def main():
    simulation = uninterruptibleSimulation.Test()
    simulation.uninterruptible('allRealistic')
    simulation.uninterruptible('realisticPredict')
    simulation.uninterruptible('allPredict')
    output = plot.Plot(simulation.testResult)
    output.power()
    output.plotUninterruptible()
    output.price()
    output.fig.tight_layout()
    output.fig.savefig('lib/plot/testing.png')

if __name__ == '__main__':
    main()