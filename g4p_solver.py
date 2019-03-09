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

from Genetic_Alghorithms import Evolution, Environment, Chromosome


def g4p_evolver(evolution, environment):
    all_results=[]

    ##-------INIT POPULATION--------##
    population = evolution.create_random_population()
    #------------------------------#
    
    for generation in range(evolution.n_generations):
        #--------------EVALUATE MODELS--------------#
        population_scores = environment.parallel_evaluate_population(population)
        population_fitness = np.mean(population_scores, axis=1)

        elite_threashold = np.mean(population_fitness)
        #------------------------------#

        print('\n ****** Generation', generation+1, 'max score = ', max(population_fitness), ' elite_threashold = ',elite_threashold,' ******\n')
        all_results.append(population)

        #-------------EXIT IF CONVERGED-------------#
        if environment.converged:
            best_policy = population[np.argmax(population_fitness)]
            break
        #------------------------------#

        #-------------NATURAL SELECTION-------------#
        elite, elite_fitness = evolution.select_elite(population, population_fitness, elite_threashold, population_scores)
        elitism=len(elite)
        #------------------------------#

        #--------------CROSSING OVER--------------# 
        offsprings = []
        el_rank=list(reversed(np.argsort(elite_fitness)))
        for i in range(elitism):
            for j in range(i+1,elitism):
                parent_A = elite[el_rank[i]]
                parent_B = elite[el_rank[j]]
                child1, child2 = evolution.crossover(parent_A, parent_B)
                child1, child2 = evolution.verify_crossover(child1, child2, offsprings)
                offsprings.append(child1)
                offsprings.append(child2)
        #------------------------------#

        #----------------MUTATION----------------#
        mutated_offsprings = [evolution.mutate(child, p=evolution.mutation_prob) for child in offsprings]    
        #------------------------------#

        #-----------NATURAL SELECTION-----------# 
        # population = elite
        population = mutated_offsprings
        print('( survived=',elitism,' childs=,', len(offsprings), ' tot_pop=', len(population),' )')
        #------------------------------#
        
    environment.pool.close()
    return environment.env, best_policy, all_results





if __name__ == '__main__':
    sid = input('Input seed for RNG    [ENTER for random seed]    ')
    if sid=='':
        sid=np.random.randint(2**32 - 1)#np.random.get_state()[1][0]
        print('using ', sid)
    else:
        sid=int(sid)

    abs_time_start = time.time()

    evolution = Evolution(
        n_chromosomes   = 20,
        n_generations   = 10,
        mutation_prob   = 0.05,
        max_elite       = 10,
        seed            = 1234,
        genotype_len    = 50,
        MAX_DEPTH       = 10
        )
    environment = Environment(
        env_id          = 'CartPole-v0',
        n_episodes      = 250,
        bins            = (3,3,3,3)
    )
    env, best_policy, all_results = g4p_evolver(evolution, environment)#123456 #2400846564
    
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
