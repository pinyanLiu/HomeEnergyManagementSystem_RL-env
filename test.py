from tensorforce import Agent,Environment
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def main():
    environment=Environment.create(environment='gym',level='Hems-v1')
    agent = Agent.load(directory = 'saver_dir',format='checkpoint',environment=environment)

    soc = []
    load = []
    pv = []
    totalReward = 0
    monthlySoc = pd.DataFrame()
    monthlyRemain = pd.DataFrame()
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
            load.append(states[1])
            pv.append(states[2])
            soc.append(states[3])
            totalReward += reward
            if i == 11:
                price.append(states[4])

        remain = [load[j]-pv[j] for j in range(len(load))]
        price = [(price[i]-np.min(price))/(np.max(price)-np.min(price)) for i in range(len(price))] #normalize price to [0,1] 
        monthlySoc.insert(i,column=str(i+1),value=soc)
       # monthlyLoad.insert(i,column=str(i+1),value=load)
       # monthlyPv.insert(i,column=str(i+1),value=pv)
        monthlyRemain.insert(i,column=str(i+1),value=remain)
        load.clear()
        pv.clear()
        soc.clear()
    print('Agent average episode reward: ', totalReward/12 )

#plot the result

    plt.rcParams["figure.figsize"] = (12.8, 9.6)
    fig,axes = plt.subplots(6,2)
    plt.title("SOC and Price for each month")
    #plt.xlabel("sampleTime")
    ax1=axes[0,0]
    ax2=axes[0,1]
    ax3=axes[1,0]
    ax4=axes[1,1]
    ax5=axes[2,0]
    ax6=axes[2,1]
    ax7=axes[3,0]
    ax8=axes[3,1]
    ax9=axes[4,0]
    ax10=axes[4,1]
    ax11=axes[5,0]
    ax12=axes[5,1]
    sub1=ax1.twinx()
    sub2=ax2.twinx()
    sub3=ax3.twinx()
    sub4=ax4.twinx()
    sub5=ax5.twinx()
    sub6=ax6.twinx()
    sub7=ax7.twinx()
    sub8=ax8.twinx()
    sub9=ax9.twinx()
    sub10=ax10.twinx()
    sub11=ax11.twinx()
    sub12=ax12.twinx()

    ax1.set_ylabel('SOC')
    ax1.plot(range(len(monthlySoc['1'][:])), monthlySoc['1'][:], label = "Jan",color='red')    
    ax1.plot(range(len(price)), price, label = "price")
    ax1.set_title('Jan')

    ax2.set_ylabel('SOC')
    ax2.plot(range(len(monthlySoc['2'][:])), monthlySoc['2'][:], label = "Feb",color='red')
    ax2.plot(range(len(price)), price, label = "price")
    ax2.set_title('Feb')

    ax3.set_ylabel('SOC')
    ax3.plot(range(len(price)), price, label = "price")
    ax3.plot(range(len(monthlySoc['3'][:])), monthlySoc['3'][:], label = "Mar",color='red')
    ax3.set_title('Mar')

    ax4.set_ylabel('SOC')
    ax4.plot(range(len(monthlySoc['4'][:])), monthlySoc['4'][:], label = "Apr",color='red')
    ax4.plot(range(len(price)), price, label = "price")
    ax4.set_title('Apr')

    ax5.set_ylabel('SOC')
    ax5.plot(range(len(monthlySoc['5'][:])), monthlySoc['5'][:], label = "May",color='red')
    ax5.plot(range(len(price)), price, label = "price")
    ax5.set_title('May')

    ax6.set_ylabel('SOC')
    ax6.plot(range(len(monthlySoc['6'][:])), monthlySoc['6'][:], label = "Jun",color='red')
    ax6.plot(range(len(price)), price, label = "price")
    ax6.set_title('Jun')

    ax7.set_ylabel('SOC')
    ax7.plot(range(len(monthlySoc['7'][:])), monthlySoc['7'][:], label = "July",color='red')
    ax7.plot(range(len(price)), price, label = "price")
    ax7.set_title('July')

    ax8.set_ylabel('SOC')
    ax8.plot(range(len(monthlySoc['8'][:])), monthlySoc['8'][:], label = "Aug",color='red')
    ax8.plot(range(len(price)), price, label = "price")
    ax8.set_title('Aug')

    ax9.set_ylabel('SOC')
    ax9.plot(range(len(monthlySoc['9'][:])), monthlySoc['9'][:], label = "Sep",color='red')
    ax9.plot(range(len(price)), price, label = "price")
    ax9.set_title('Sep')

    ax10.set_ylabel('SOC')
    ax10.plot(range(len(monthlySoc['10'][:])), monthlySoc['10'][:], label = "Oct",color='red')
    ax10.plot(range(len(price)), price, label = "price")
    ax10.set_title('Oct')

    ax11.set_ylabel('SOC')
    ax11.plot(range(len(monthlySoc['11'][:])), monthlySoc['11'][:], label = "Nov",color='red')
    ax11.plot(range(len(price)), price, label = "price")
    ax11.set_title('Nov')

    ax12.set_ylabel('SOC')
    ax12.plot(range(len(monthlySoc['12'][:])), monthlySoc['12'][:], label = "Dec",color='red')
    ax12.plot(range(len(price)), price, label = "price")
    ax12.set_title('Dec')

    sub1.set_ylabel('Power')
    sub1.plot(range(len(monthlyRemain['1'][:])), monthlyRemain['1'][:],color='green')  
    sub1.plot(range(len(monthlyRemain['1'][:])), np.zeros(95),'--')

    sub2.set_ylabel('Power')
    sub2.plot(range(len(monthlyRemain['2'][:])), monthlyRemain['2'][:],color='green')  

    sub3.set_ylabel('Power')
    sub3.plot(range(len(monthlyRemain['3'][:])), monthlyRemain['3'][:],color='green')  

    sub4.set_ylabel('Power')
    sub4.plot(range(len(monthlyRemain['4'][:])), monthlyRemain['4'][:],color='green')  

    sub5.set_ylabel('Power')
    sub5.plot(range(len(monthlyRemain['5'][:])), monthlyRemain['5'][:],color='green')  

    sub6.set_ylabel('Power')
    sub6.plot(range(len(monthlyRemain['6'][:])), monthlyRemain['6'][:],color='green')  

    sub7.set_ylabel('Power')
    sub7.plot(range(len(monthlyRemain['7'][:])), monthlyRemain['7'][:],color='green')  

    sub8.set_ylabel('Power')
    sub8.plot(range(len(monthlyRemain['8'][:])), monthlyRemain['8'][:],color='green')  

    sub9.set_ylabel('Power')
    sub9.plot(range(len(monthlyRemain['9'][:])), monthlyRemain['9'][:],color='green')  

    sub10.set_ylabel('Power')
    sub10.plot(range(len(monthlyRemain['10'][:])), monthlyRemain['10'][:],color='green')  

    sub11.set_ylabel('Power')
    sub11.plot(range(len(monthlyRemain['11'][:])), monthlyRemain['11'][:],color='green')  

    sub12.set_ylabel('Power')
    sub12.plot(range(len(monthlyRemain['12'][:])), monthlyRemain['12'][:],color='green')  

    fig.tight_layout()
    fig.savefig('pic/plot.png')

    network = agent.get_architecture()
    print(network)
    # Close agent and environment
    agent.close()
    environment.close()

if __name__ == '__main__':
    main()