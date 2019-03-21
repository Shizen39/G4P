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

from anytree.exporter import DotExporter
import os, shutil

from Genetic_Gym import Population, Environment



def evolve(population, environment, initial_n_chr, n_generations, genotype_len, seed, MAX_DEPTH, MAX_WRAP=2):
    np.random.seed(seed)
    environment.seed = seed

    all_populations=[]

    ##-------INIT POPULATION--------##
    # get initial chromosomes generated by the set of genotype 
    population.initialize_chromosomes(initial_n_chr, genotype_len, MAX_DEPTH, MAX_WRAP)
    pool = Pool(multiprocessing.cpu_count())
    #------------------------------#
    last_max_fitness=None
    ctr=0
    for generation in range(n_generations):
        #--------------EVALUATE MODELS--------------#
        population.chromosomes_scores   = environment.parallel_evaluate_population(population, pool, to_file=False)

        population.chromosomes_fitness  = np.mean(population.chromosomes_scores, axis=1)
        #------------------------------#

        
        #-------------EXIT IF CONVERGED-------------#
        
        
        print('\n ****** Generation', generation+1, 'max score = ', max(population.chromosomes_fitness) , ' elite_threashold = ',np.mean(population.chromosomes_fitness),' ******\n')
        
        population.best_individual = population.chromosomes[np.argmax(population.chromosomes_fitness)]
        all_populations.append(population)
        if environment.converged or generation==n_generations-1:
            break
        #------------------------------#
       
        
                
        #-------------NATURAL SELECTION-------------#
        population.survival_threashold  = np.mean(population.chromosomes_fitness)

        # for i,chromosome in enumerate(population.chromosomes):
        #     if population.chromosomes_fitness[i]>=population.survival_threashold:
        #         chromosome.tree_to_png(generation)
        #         chromosome.generate_solution(generation, to_file=True)

        estingued, estingued_fitness = population.do_natural_selection()
        
        elites_len = len(population.chromosomes)
        #------------------------------#


        #--------------CROSSING OVER--------------# 
        
        ranks = list(reversed(np.argsort(population.chromosomes_fitness)))

        select_probs = np.array(population.chromosomes_fitness) / np.sum(population.chromosomes_fitness)
        if np.sum(population.chromosomes_fitness) <0:
            select_probs = select_probs[::-1]


        if np.max(population.chromosomes_fitness) == last_max_fitness:  
            ctr+=1
            if ctr>=1:
                population.chromosomes = [population.mutate(child, leaves_only=True) for child in population.chromosomes] 
        #         unique_fit=[]
        #         unique_chr=[]
        #         for i,fit in enumerate(population.chromosomes_fitness):
        #             if unique_fit.count(fit)<=2:
        #                 unique_fit.append(fit)
        #                 unique_chr.append(population.chromosomes[i])
                
        #         population.chromosomes=unique_chr
        #         population.chromosomes_fitness=unique_fit

        #         fix = population.max_elite - len(population.chromosomes)
        #         population.chromosomes+=estingued[-fix:]
        #         population.chromosomes_fitness+=estingued_fitness[-fix:]

        #         ranks = list(reversed(np.argsort(population.chromosomes_fitness)))
        #         elites_len = len(population.chromosomes)
                
        #         for _ in range(ctr):
        #             print('mutating')
        #             population.chromosomes = [population.mutate(child) for child in population.chromosomes]
        else:
            ctr=0
        last_max_fitness = max(population.chromosomes_fitness)

        print('crossing-over...')
        offsprings = []
        jobs=[]
        dk = int(initial_n_chr/2)
        random_seeds=[np.random.randint(2**32 - 1) for i in range(dk)]
        population.chromosomes= np.array(population.chromosomes)
        parents = [population.chromosomes[np.random.choice(range(elites_len), 2, p=select_probs)] 
                    for _ in range(dk)]
        for i,parent in enumerate(parents):
            jobs.append(pool.apply_async(population.crossover, [parent[0], parent[1], random_seeds[i]]))
        # for i in range(elites_len):
        #     seed_i=[]
        #     for j in range(i+1, elites_len):
        #         seed_i.append(np.random.randint(2**32 - 1))
        #     random_seeds.append(seed_i)

        # for i in range(elites_len):
        #     for j in range(i+1,elites_len):
        #         # print('crossingover ', population.chromosomes[ranks[i]].cid, population.chromosomes[ranks[j]].cid)
        #         jobs.append(pool.apply_async(population.crossover, [
        #                     population.chromosomes[ranks[i]], population.chromosomes[ranks[j]],
        #                     random_seeds[i][j-i-1]
        #                     ]))
        for j in jobs:
            child1,child2=j.get()
            offsprings.append(child1)
            offsprings.append(child2)        
        #------------------------------#

        #----------------MUTATION----------------#
        print('mutating..')    
        mutated_offsprings = [population.mutate(child) for child in offsprings]    
           

        #------------------------------#

        #-----------NEXT GENERATION-----------# 
        # population = elite
        population = Population(mutation_prob=population.mutation_prob, crossover_prob=population.crossover_prob, max_elite=population.max_elite, environment=environment)
        population.chromosomes = mutated_offsprings
        print('( childs=', len(offsprings), ' tot_pop=', len(population.chromosomes),' )\n\n')
        #------------------------------#
        
    pool.close()
    return all_populations





if __name__ == '__main__':
    if os.path.exists('./outputs'):
        shutil.rmtree('./outputs')
    os.mkdir('./outputs')

    sid = input('Input seed for RNG    [ENTER for default, r for random]    ')
    if sid=='':
        sid=2468609729 #1234
    if sid=='r':
        sid=np.random.randint(2**32 - 1)
        print('using ', sid)
    else:
        sid=int(sid)

    abs_time_start = time.time()

    environment = Environment(
            env_id          = 'CartPole-v0',
            n_episodes      = 100,
            bins            = (6,3,6,5)
        )
    population = Population(
        mutation_prob   = 0.9,
        crossover_prob  = 0.9,
        max_elite       = 12,
        environment     = environment
    )
    all_populations = evolve(
        population, 
        environment, 
        initial_n_chr = 185, 
        n_generations = 5,
        seed          = sid,
        genotype_len  = 22,
        MAX_DEPTH     = 5,
        MAX_WRAP=3
    )

    # environment = Environment( 
    #         env_id          = 'MountainCar-v0', # 1. prova con seed diversi
    #         n_episodes      = 100,
    #         bins            = (10,10) # 2. ho provato 9,10 e 10,9 ma danno meno di 116 (CON SEED 1234 !!!!!! INSERISCILO A MANO)
    #     )
    # population = Population(
    #     mutation_prob   = 1.0,
    #     crossover_prob  = 1.0,
    #     max_elite       = 26, # 3. 27 no. 26 (115),  25 E 23 mi ha dato -116.85 (CHR 387 GEN 3, then not converged enymore) -> prova a diminuire max_elite
    #     environment     = environment
    # )
    # all_populations = evolve(
    #     population, 
    #     environment, 
    #     initial_n_chr = 300, # 4.  250 sì(116), 300 no (115)!!!
    #     n_generations = 8,
    #     seed          = sid,
    #     genotype_len  = 20, # 5. questi 
    #     MAX_DEPTH     = 5, # 5.poi cambia questi lasciando tutto invariato
    #     MAX_WRAP      = 2 # 5.questi
    # )


    abs_time= time.time() - abs_time_start
    
    #---------------plotting-------------#
    print('Plotting ... ')


    for generation, population in enumerate(all_populations):
        population.best_individual.tree_to_png(generation)
        population.best_individual.generate_solution(generation, to_file=True)

    ep_len = len(all_populations[0].chromosomes_scores[0])
    z_axys = np.arange(ep_len)
    for i,population in enumerate(all_populations):
        ax= plt.figure(figsize=(20, 19)).add_subplot(111, projection='3d')
        best_idx = np.argmax(population.chromosomes_fitness)
        if len(population.chromosomes_scores)>12:
            low =  0 if best_idx-5<0 else best_idx-10 if best_idx+5>=len(population.chromosomes_scores) else best_idx-5
            high = len(population.chromosomes_scores) if best_idx+5>=len(population.chromosomes_scores)-1 else best_idx+5
            scores = np.array(population.chromosomes_scores)[range(low, high)] 
        else:
            scores = population.chromosomes_scores
        ax.set_xticks( np.arange(len(scores)) )
        for j,score in enumerate(scores):
            ax.plot(np.full(ep_len, j, int)  , z_axys, score, zorder=j)

        ax.set_zlabel("Rewards")
        ax.set_ylabel("Episode")
        ax.set_xlabel("Chromosome")
        
        title=  environment.env.spec.id+" solved in {} generations\n".format(len(all_populations)-1)
        title += "time elapsed = {} sec\n".format(abs_time)
        title += "GENERATION [ {} / {} ]".format(i, len(all_populations)-1)
        plt.title(title)
        save_dir = './outputs/GEN-{}/'.format(i)
        plt.savefig(save_dir+'plot.png', bbox_inches='tight')
    print('used seed = ', sid)
    #--------------evaluate--------------------#
    wrap = input('Do you want to run the evolved policy and save it?    [y/N]    ')
    if wrap=='y':
        import os
        save_dir = './outputs/'+environment.env.spec.id+'_results/' + str(time.time()) + '/'
        # env.seed(0)
        environment.env = wrappers.Monitor(environment.env, save_dir, force=True)
        best_policy = all_populations.pop().best_individual
        for episode in range(ep_len):
            environment.run_one_episode(environment.env, best_policy, episode, prnt=True)
        environment.env.env.close()
    else:
        environment.env.close()
    
