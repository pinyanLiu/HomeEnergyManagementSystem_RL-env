from tensorforce import Agent,Environment
import pandas as pd
import matplotlib.pyplot as plt

def main():
    environment=Environment.create(environment='gym',level='Hems-v1')
    agent = Agent.load(directory = 'saver_dir',format='checkpoint',environment=environment)

    soc = []
    monthlySoc = pd.DataFrame()
    price = []
    for i in range(12):
        states = environment.reset()
        internals = agent.initial_internals()
        terminal = False
        while not terminal:
            actions, internals = agent.act(
                states=states, internals=internals, independent=True, deterministic=True
            )
            states, terminal, reward = environment.execute(actions=actions)
            soc.append(states[3])
            if i == 11:
                price.append(states[4])
        monthlySoc.insert(i,column=str(i),value=soc)
        soc.clear()
        

    
    fig,sub1 = plt.subplots()
    #fig = plt.figure()
    plt.title("SOC")
    plt.xlabel("sampleTime")
    sub2=sub1.twinx()
    sub1.set_ylabel('SOC')
    sub1.plot(range(len(monthlySoc['10'][:])), monthlySoc['1'][:], label = "soc",color='red')
    sub2.set_ylabel('Price')
    sub2.plot(range(len(price)), price, label = "price")
    fig.tight_layout()
    fig.savefig('pic/plot.png')

    # Close agent and environment
    agent.close()
    environment.close()

if __name__ == '__main__':
    main()