from lib.plot import plot
from lib.test import test

if __name__ == '__main__':
    simulation = test.Test()
    simulation.uninterruptible()
    output = plot.Plot(simulation.testResult)
    output.power()
    output.fig.tight_layout()
    output.fig.savefig('lib/plot/testing.png')
