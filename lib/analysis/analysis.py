
def calScheduleSimilarity(schedule1,schedule2):
    assert len(schedule2)==len(schedule1) , 'len of two argument not equal'
    result = 0
    for i in range(len(schedule1)):
        if schedule1[i]==schedule2[i]:
            result+=1
    return result/len(schedule1)

def calMAPE():
    pass