import numpy as np
import time
import gym
import gym.wrappers as wrappers
import gym.spaces as spaces
from collections import deque
import matplotlib.pyplot as plt         
from mpl_toolkits.mplot3d import Axes3D
from multiprocessing import Pool
import multiprocessing
#from joblib import Parallel, delayed

global space_bins
global converged
global env

def discretize_space(space): #generic func to assign a discrete state from a space divided in intervalls
    global space_bins
    space_len= space.__len__()
    new_space = []
    for i in range(space_len):
        new_space.append(np.digitize([space[i]], space_bins[i])[0])    
    return tuple(new_space)
def binning(obs_space, bins): #generic func to discretize a continous space
    sp_low, sp_up = obs_space.low, obs_space.high
    inf=np.finfo(np.float32).max
    div_inf=7000**10
    sp_intervals=[]
    for i in range(bins.__len__()):
        if sp_low[i] == -np.inf:
                sp_low[i] = -inf
        if sp_up[i] == np.inf:
            sp_up[i] = inf
        sp_intervals.append([sp_low[i]/div_inf if sp_low[i]==-inf else sp_low[i], sp_up[i]/div_inf if sp_up[i]==inf else sp_up[i]])
    ret_bins=[]
    for i,v in enumerate(sp_intervals):
        x= np.histogram(v, bins[i])[1]    #subdivide continous interval into equal spaced bins[i] intervals
        ret_bins.append(x)
    return ret_bins     #return: Array of binned intervals; e.g. [[i,..],[i,..],..]


def run_one_episode(env, policy, episode, prnt=False, render=False):
    episode_reward = 0
    done = False
    obs = env.reset()
    while not done:
        if render: env.render()
        obs = discretize_space(obs)
        action = policy[obs]
        obs, reward, done, _ = env.step(action)
        episode_reward += reward
    if prnt: print('V' if episode_reward == 200 else 'X'," Ep. ",episode," terminated (", episode_reward, "rewards )")
    return episode_reward

#----FITNESS-----#
def evaluate_policy(env, policy, n_episodes, i,prnt=False, render=False):
    env.seed(0)
    total_rewards = deque(maxlen = env.spec.trials)
    for episode in range(n_episodes):
        reward = run_one_episode(env, policy, episode, prnt, render)
        total_rewards.append(reward)
        if np.mean(total_rewards) >= env.spec.reward_threshold and episode>=env.spec.trials: #getting reward of 195.0 over 100 consecutive trials
            break 
    print("Chromosome ",i,"fitness = ",np.mean(total_rewards))
    return list(total_rewards)

#----CROSSING OVER-----#
def crossover(a,b,elite,fitness,n_chromosomes,select_probs):
    print('~ crossing over ', fitness[a],' and ', fitness[b])
    policy_A = elite[a]
    policy_B = elite[b]
    child_policy1 = dict()
    child_policy2 = dict()
    cross_weight = abs(env.spec.reward_threshold / fitness[a])    #percentage of childs' genes eredited by a
    split=int(len(policy_A)*cross_weight)
    tmp = list(policy_A.values())[:split]+list(policy_B.values())[-(len(policy_B)-split):]
    for idx,k in enumerate(elite[a]):
        child_policy1[k]=tmp[idx] 
    tmp = list(policy_B.values())[:len(policy_B)-split]+list(policy_A.values())[-split:]
    for idx,k in enumerate(elite[a]):
        child_policy2[k]=tmp[idx] 
    return child_policy1, child_policy2

#----MUTATE-----#
def mutate(child, p=0.05):
    mutated = child
    for k in mutated:
        if np.random.uniform() < p:
            mutated[k] = np.random.randint(0,env.action_space.n)
    return mutated

#----INIT POPULATION-----#
def create_random_population(bins, n_chromosomes):
    import itertools
    ranges=[range(v+2) for v in bins]
    states = []
    for r in ranges:
        states.append([v for v in r])
    comb_states= list(itertools.product(*states))

    population=[]   
    for _ in range(n_chromosomes):
        chromosome = {} #{(state):action, ...., (state):action}
        for s in comb_states:
            chromosome[s]= np.random.randint(0,env.action_space.n)
        population.append(chromosome)
    return population   #[chromosome, ..., chromosome]

#----PARALLELIZE EVALUATION-----#
def run_parallel_eval(env_id, sid, pool, n_episodes, population):
    global converged
    pop_scores = [] 
    jobs=[]
    for i,p in enumerate(population):                                           #pop_scores = Parallel(n_jobs=-1)(delayed(evaluate_policy)(env, p, n_episodes) for p in population if not converged)
        jobs.append(pool.apply_async(evaluate_policy, [env, p, n_episodes, i]))
    for j in jobs:
        if not converged:
            if not j.ready():
                j.wait()
            score=j.get()
            pop_scores.append(score)
            if np.mean(score)>=env.spec.reward_threshold:
                converged = True
        else:
            pool.terminate()
    return pop_scores


#----START EVOLVING-----#
def evolve(env_id, n_chromosomes, n_generations, bins, sid=1234, n_episodes=250, mut_prob=0.05, max_elite=5): #20 50 5 250
    global space_bins
    global converged
    global env

    np.random.seed(sid)
    env = gym.make(env_id)
    env.seed(0)

    #-------INITIALIZATION--------#
    space_bins = binning(env.observation_space, bins)
    population = create_random_population(bins, n_chromosomes)
    all_results=[]
    converged=False
    pop_scores = []
    max_fit_tmp=np.inf
    pool = Pool()
    ct=0
    for generation in range(n_generations): #meiosi
        print('\n ************** GENERATION', generation+1, '**************\n')
        #-------EVALUATE MODELS--------#
        pop_scores = run_parallel_eval(env_id, sid, pool, n_episodes, population)   
        pop_fitness = np.mean(pop_scores, axis=1) #mean of all episode_rewards for each policy

        elite_threashold = np.mean(pop_fitness)
        print('\n ****** Generation', generation+1, 'max score = ', max(pop_fitness), ' elite_threashold = ',elite_threashold,' ******\n')
        
        #------EXIT IF CONVERGED--------#
        if len(pop_scores)<=30:
            all_results.append(pop_scores)
        else:
            tmp_pop_scores = [p for p in pop_scores if np.mean(p)>=elite_threashold]
            new_pop_scores = [list(p) for p in set(tuple(x) for x in tmp_pop_scores) ]
            all_results.append(new_pop_scores)
        if ct==1 and max_fit_tmp == max(pop_fitness):
            all_results.pop()
            best_policy = population[np.argmax(pop_fitness)]
            break 
        if converged:
            best_policy = population[np.argmax(pop_fitness)]
            break
        
        #-------ELITISM (SELECT FITNEST MODELS)--------#
        elite = [e for i,e in enumerate(population) if pop_fitness[i]>=elite_threashold] #survive only those fitness is greater then  mean of all fitness
        elite_score = [e for i,e in enumerate(pop_scores) if pop_fitness[i]>=elite_threashold]
        elite_fitness = list(np.mean(elite_score, axis=1))

        if len(elite)>max_elite:
            while len(elite)>max_elite:
                rm= np.argmin(elite_fitness)
                elite.pop(rm)
                elite_score.pop(rm)
                elite_fitness.pop(rm)
        elitism=len(elite)

        #-------CROSSOVER (REPRODUCTION)--------#            
        select_probs = np.array(elite_fitness) / np.sum(elite_fitness)
    
        offsprings = []
        el_rank=list(reversed(np.argsort(elite_fitness))) 
        for i in range(elitism):
            for j in range(i+1,elitism):
                a=el_rank[i]
                b=el_rank[j]

                c1,c2=crossover(a,b, elite, elite_fitness, elitism, select_probs)
                if c1 not in offsprings:
                    offsprings.append(c1)
                else:
                    while c1 in offsprings:
                        k_idx=np.random.randint(len(c1.keys()))
                        k=list(c1.keys())
                        c1[k[k_idx]] = np.random.randint(env.action_space.n)
                offsprings.append(c1)
                if c2 not in offsprings:
                    offsprings.append(c2)
                else:
                    while c2 in offsprings:
                        k_idx=np.random.randint(len(c2.keys()))
                        k=list(c2.keys())
                        c2[k[k_idx]] = np.random.randint(env.action_space.n)
                offsprings.append(c2)
        
        #-------MUTATION--------#
        mutated_offsprings = [mutate(p, p=mut_prob) for p in offsprings]    

        #-------POPULATION REPLACEMENT (FITNEST SURVIVAL)--------#
        # population = elite
        population = mutated_offsprings
        if max_fit_tmp == max(pop_fitness):
            ct+=1
        max_fit_tmp = max(pop_fitness)

        print('( survived=',elitism,' childs=,', len(offsprings), ' tot_pop=', len(population),' )')
    
    pool.close()
    return env, best_policy, all_results


if __name__ == '__main__':
    sid = input('Input seed for RNG    [ENTER for random seed]    ')
    if sid=='':
        sid=np.random.randint(2**32 - 1)#np.random.get_state()[1][0]
        print('using ', sid)
    else:
        sid=int(sid)

    abs_time_start = time.time()
    
    env, best_policy, all_results = evolve('CartPole-v0', 20, 50, (1,1,12,5), sid=sid)#123456 #2400846564
    
    # env, best_policy, all_results = evolve('MountainCar-v0', 200, 50, (7,2), sid=sid, mut_prob=0.17, max_elite=11)#333555669

    abs_time= time.time() - abs_time_start
    
    #---------------plotting-------------#
    fig = plt.figure()
    ep_len=all_results[0][0].__len__()
    z=np.arange(ep_len)
    for i,v in enumerate(all_results):
        ax= fig.add_subplot(all_results.__len__(), 1, i+1, projection='3d')
        ax.set_xticks( np.arange(len(v)) )
        for j,pol in enumerate(v):
            ax.plot(np.full(ep_len, j, int), z,  pol, zorder=j)
        ax.set_zlabel("Rewards of gen %s"%(i+1))
        ax.set_ylabel("Episode")
        ax.set_xlabel("Chromosome")
        if i==0:
            title=  env.spec.id+" solved in "+ str(len(all_results))+" generations\n"
            title += "absolute time elapsed = "+str(abs_time)+"sec\n"
            plt.title(title)
            ax.legend()
    plt.show()

    #--------------evaluate--------------------#
    wrap = input('Do you want to run the evolved policy and save it?    [y/N]    ')
    if wrap=='y':
        import os
        save_dir = os.path.dirname(os.path.abspath(__file__))+'/'+env.spec.id+'_results/' + str(time.time()) + '/'
        env = wrappers.Monitor(env, save_dir, force=True)
        for episode in range(ep_len):
            run_one_episode(env, best_policy, episode, True)
        env.env.close()
        plt.savefig(save_dir+'plot.png', bbox_inches='tight')
    else:
        env.close()
    print('used seed = ', sid)
