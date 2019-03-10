'''
This is the main execution files that use class utilities from Genetic_Gym.py

In particular it defines:
- evolve() function that describe the population flow (init, evaluate, select, crossingover, mutate, ...)
- main() function execute evolve() using parametrized Genetic_Gym.Population and Genetic_Gym.Environment,
plotting all single generation chromosomes and their population informations in multiple graphs
and finally (and eventually) showing the evolved chromosome in action
'''


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

from Genetic_Gym import Population, Environment

#TODO: crossover

def evolve(population, environment, n_chromosomes, n_generations):
    all_results=[]
    genotype_len = 50
    MAX_DEPTH = 10
    ##-------INIT POPULATION--------##
    # get initial chromosomes generated by the set of genotype 
    population.initialize_chromosomes(n_chromosomes, genotype_len, MAX_DEPTH)
    pool = Pool()
    #------------------------------#
    
    for generation in range(n_generations):
        #--------------EVALUATE MODELS--------------#
        population.chromosomes_scores   = environment.parallel_evaluate_population(population, pool)
        population.chromosomes_fitness  = np.mean(population.chromosomes_scores, axis=1)
        
        #------------------------------#

        #-------------EXIT IF CONVERGED-------------#
        population.best_policy = population.chromosomes[np.argmax(population.chromosomes_fitness)]
        print('\n ****** Generation', generation+1, 'max score = ', max(population.chromosomes_fitness), ' elite_threashold = ',population.survival_threashold,' ******\n')
        all_results.append(population)
        if environment.converged:
            break
        #------------------------------#

        #-------------NATURAL SELECTION-------------#
        population.survival_threashold  = np.mean(population.chromosomes_fitness)

        population.do_natural_selection()
        
        elites_len = len(population.chromosomes)
        #------------------------------#

        #--------------CROSSING OVER--------------# 
        ranks = list(reversed(np.argsort(population.chromosomes_fitness)))
        offsprings = []
        for i in range(elites_len):
            for j in range(i+1,elites_len):
                child1, child2 = population.crossover(
                    population.chromosomes[ranks[i]],
                    population.chromosomes[ranks[j]]
                )
                #child1, child2 = population.verify_crossover(child1, child2, offsprings)
                offsprings.append(child1)
                offsprings.append(child2)
        #------------------------------#

        #----------------MUTATION----------------#
        mutated_offsprings = [population.mutate(child, p=population.mutation_prob) for child in offsprings]    
        #------------------------------#

        #-----------NEXT GENERATION-----------# 
        # population = elite
        population = Population(mutation_prob=population.mutation_prob, max_elite=population.max_elite, seed=population.seed)
        population.chromosomes = mutated_offsprings
        print('( survived=',elites_len,' childs=,', len(offsprings), ' tot_pop=', len(population.chromosomes),' )')
        #------------------------------#
        
    environment.pool.close()
    return environment.env, all_results





if __name__ == '__main__':
    sid = input('Input seed for RNG    [ENTER for random seed]    ')
    if sid=='':
        sid=np.random.randint(2**32 - 1)#np.random.get_state()[1][0]
        print('using ', sid)
    else:
        sid=int(sid)

    abs_time_start = time.time()

    population = Population(
        mutation_prob   = 0.05,
        max_elite       = 10,
        seed            = 1234,
    )
    environment = Environment(
        env_id          = 'CartPole-v0',
        n_episodes      = 250,
        bins            = (3,3,3,3)
    )

    env, all_results = evolve(
        population, 
        environment, 
        n_chromosomes = 20, 
        n_generations = 10
    )#123456 #2400846564
    
    # env, best_policy, all_results = evolve('MountainCar-v0', 200, 50, (7,2), sid=sid, mut_prob=0.17, max_elite=11)#333555669

    abs_time= time.time() - abs_time_start
    
    #---------------plotting-------------#
    #TODO: PLOT EVERY GENERATIO IN DISTINCT FILES (TITLE= GEN [1/N])
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

        best_policy = all_results.pop().best_individual
        for episode in range(ep_len):
            environment.run_one_episode(env, best_policy, episode, True)
        env.env.close()
        plt.savefig(save_dir+'plot.png', bbox_inches='tight')
    else:
        env.close()
    print('used seed = ', sid)
