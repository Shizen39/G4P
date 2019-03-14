from multiprocessing import Pool
import time
import sys, os
import numpy as np

def f( sid, t):
    # np.random.seed(sid)
    # print('seed=',np.random.get_state()[1][0])
    rng = np.random.RandomState()
    
    rng.seed(sid)
    # print(list(rng.get_state()[1])==list(np.random.get_state()[1]))
    time.sleep(t)
    return rng.randint(1,50, size=10)



pool = Pool()

sleep_times = np.random.uniform(0,4, size=10)

np.random.seed(0)



t1= time.time()
for k in range(1,4):
    jobapply=[]
    retapply=[]
    for i in range(10):
        # print("np\n",np.random.get_state())
        jobapply.append(pool.apply_async(f, [np.random.randint(999999),sleep_times[i]]))
    for j in jobapply:
        if not j.ready():
            # print(j)
            j.wait()
        res=j.get()
        print(res)
        retapply.append(res)
    # print("apply:\n",retapply)
    print("TIME = ", time.time()-t1, 'over ', np.sum(sleep_times))
    

