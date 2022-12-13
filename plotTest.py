from lib.plot import plot
from lib.test import test


def main():
    simulation = test.Test()
    simulation.uninterruptible()
    print(type(simulation.testResult[0]))
    print(type(simulation.testResult[0]['switch']))
    output = plot.Plot(simulation.testResult)
    output.power()
    output.uninterruptibleLoad()
    output.fig.tight_layout()
    output.fig.savefig('lib/plot/testing.png')

if __name__ == '__main__':
    main()