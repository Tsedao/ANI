import numpy as np


def add_temporal_noise(day, dt=1):
    return day + np.random.rand(*day.shape)*dt

def event2count(data: np.ndarray, dt:int =1)-> np.ndarray :
    """convert event data to count data
    """
    max_t = int(np.max(data[:,0]))
    num_type = int(np.max(data[:,1]) + 1)
    dt = dt
    counts = np.zeros(shape=(max_t, num_type))

    ## iterate over T
    for i in range(0,max_t,dt):
        ## iterate over history event
        for j in range(len(data)):
            ## mark the event if the event happen before time i, accumlate all previous counts
            if data[j,0] <= i:
                counts[i,int(data[j,1])] += 1
            ## jump out the loop since there are no data in history smaller than i
            else:
                break
    ## calculate increased event
    final_data = np.diff(counts,axis=0)

    return final_data


