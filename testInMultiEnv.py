    def __testInHvacSoc__(self):
        self.hvacEnv = Environment.create(environment = 'gym',level='Hems-v7')
        self.socEnv = Environment.create(environment = 'gym',level='Hems-v1')
        self.hvacAgent = Agent.load(directory = 'HVAC/saver_dir',environment=self.hvacEnv)
        self.socAgent = Agent.load(directory = 'Soc/saver_dir',environment=self.socEnv)
        load = []
        hvac = []  
        soc = []
        socPower = []
        pv = []
        indoorTemperature = []
        outdoorTemperature = []
        userSetTemperature = []
        totalReward = 0
        self.monthlySoc = pd.DataFrame()
        self.monthlyIndoorTemperature = pd.DataFrame()
        self.monthlyOutdoorTemperature = pd.DataFrame()
        self.monthlyRemain = pd.DataFrame()
        self.monthlyHVAC = pd.DataFrame()
        self.monthlyUserSetTemperature = pd.DataFrame()
        self.price = []
        for month in range(12):
            states = self.hvacEnv.reset()
            hvacInternals = self.hvacAgent.initial_internals()
            socInternals = self.socAgent.initial_internals()
            terminal = False
            while not terminal:
                actions, hvacInternals = self.hvacAgent.act(
                    states=states, internals=hvacInternals, independent=True, deterministic=True
                )
                states, terminal, reward = self.hvacEnv.execute(actions=actions)

                actions, socInternals = self.socAgent.act(
                    states=states, internals=socInternals, independent=True, deterministic=True
                )
                states, terminal, reward = self.hvacEnv.execute(actions=actions)




                load.append(states[1])
                pv.append(states[2])
                if month == 11:
                    self.price.append(states[3])
                indoorTemperature.append(states[4])
                outdoorTemperature.append(states[5])
                userSetTemperature.append(states[6])
                hvac.append(actions[0])
                totalReward += reward

            remain = [load[sampletime]-pv[sampletime] for sampletime in range(95)]
            #store testing result in each dictionary
            self.monthlySoc.insert(month,column=str(month+1),value=soc)
            self.monthlySocPower.insert(month,column=str(month+1),value=socPower)
            self.monthlyIndoorTemperature.insert(month,column=str(month+1),value=indoorTemperature)
            self.monthlyOutdoorTemperature.insert(month,column=str(month+1),value=outdoorTemperature)
            self.monthlyRemain.insert(month,column=str(month+1),value=remain)
            self.monthlyHVAC.insert(month,column=str(month+1),value=hvac)
            self.monthlyUserSetTemperature.insert(month,column=str(month+1),value=userSetTemperature)
            load.clear()
            pv.clear()
            soc.clear()
            indoorTemperature.clear()
            outdoorTemperature.clear()
            userSetTemperature.clear()
            hvac.clear()
        print('Agent average episode reward: ', totalReward/12 )