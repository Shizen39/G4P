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
    x=3
    for generation in range(n_generations):
        #--------------EVALUATE MODELS--------------#
        if population.mutation_prob<0:
            population.mutation_prob=0.
        n = len(population.chromosomes)

        population.chromosomes_scores   = environment.parallel_evaluate_population(population, pool, to_file=False, prnt=True)

        population.chromosomes = [population.chromosomes[i] for i,score in enumerate(population.chromosomes_scores) if score!=None]
        population.chromosomes_scores = [score for score in population.chromosomes_scores if score!=None]

        population.chromosomes_fitness  = np.mean(population.chromosomes_scores, axis=1)
        #------------------------------#
        
        
        #-------------EXIT IF CONVERGED-------------#
        print('\n ****** Generation', generation+1, 'max score = ', max(population.chromosomes_fitness) , ' survival_threashold = ',np.mean(population.chromosomes_fitness),' ******\nDied = ',n - len(population.chromosomes),'\n')
        # population.fitness_share()
        # print(population.chromosomes_fitness)
        print(population.chromosomes_fitness)

        population.best_individual = population.chromosomes[np.argmax(population.chromosomes_fitness)]
        all_populations.append(population)


        population.best_individual.generate_solution(-1,True)
        population.best_individual.tree_to_png(-1)


        if environment.converged or generation==n_generations-1:
            break
        #------------------------------#
        # print(population.chromosomes_fitness)
        
       
                
        #-------------NATURAL SELECTION-------------#
        population.survival_threashold  = np.mean(population.chromosomes_fitness)

        population.do_natural_selection(True)
        if len(population.chromosomes)<population.max_elite:
            print('fixing....')
            n_new_chr = population.max_elite - len(population.chromosomes)
            new_pop= Population(population.mutation_prob, population.crossover_prob, population.max_elite, environment)
            new_pop.initialize_chromosomes(n_new_chr, genotype_len, MAX_DEPTH, MAX_WRAP)
            new_pop.chromosomes_scores = environment.parallel_evaluate_population(new_pop, pool, to_file=False, prnt=False)
            new_pop.chromosomes_fitness = np.mean(new_pop.chromosomes_scores, axis=1)
            population.chromosomes = list(population.chromosomes) + list(new_pop.chromosomes)
            population.chromosomes_fitness = np.array(list(population.chromosomes_fitness) + list(new_pop.chromosomes_fitness))
        elif len(population.chromosomes)>population.max_elite:
            population.do_natural_selection(False)
        print("Survived:\n",population.chromosomes_fitness)
        #------------------------------#


        #--------------CROSSING OVER--------------# 
        if np.max(population.chromosomes_fitness) == last_max_fitness:  
            ctr+=1
            last_max_fitness = np.max(population.chromosomes_fitness)
            population.mutation_prob+=(population.mutation_prob/n_generations)
            if ctr>=1:
                print('hardly mutating......', ctr)
                if ctr>=2:
                    for _ in range(ctr):
                        population.chromosomes = [population.mutate(c, np.random.randint(10), inverse_prob=True)
                    if population.chromosomes_fitness[i]==last_max_fitness else c for i,c in enumerate(population.chromosomes)]
                if ctr >=3:
                    population.fitness_share()
                    print("Shared:\n",population.chromosomes_fitness)

                    
                    # x+=1
                    # x = 2*x-1
                    # if ctr>2:
                    # print(population.chromosomes_fitness)
                    # population.fitness_share()
                    # print(population.chromosomes_fitness)

                    # population.chromosomes = [c for i,c in enumerate(population.chromosomes) if population.chromosomes_fitness[i]!=last_max_fitness]
                    # population.chromosomes_fitness = [f for f in population.chromosomes_fitness if f!=last_max_fitness]
                    
                    population.chromosomes = [population.mutate(c, leaves_only=True) for i,c in enumerate(population.chromosomes)]
                    print(population.chromosomes_fitness)
                    
                    
               
                    # population.chromosomes = [population.mutate(c, leaves_only=True, p=0.7) for i,c in enumerate(population.chromosomes)]
                # if ctr==1:    
                #     population.chromosomes = [population.mutate(c, leaves_only=True, p=0.3) for i,c in enumerate(population.chromosomes)]
        else:
            if ctr!=0 and max(population.chromosomes_fitness)>last_max_fitness:
                bid = np.argmax(population.chromosomes_fitness)
                population.chromosomes_fitness[bid] = population.chromosomes_fitness[bid]*2 if population.chromosomes_fitness[bid]>0 else population.chromosomes_fitness[bid]/2
            last_max_fitness = np.max(population.chromosomes_fitness)
            ctr=0
            x=1
        
        
       

        elites_len = len(population.chromosomes)
        select_probs = np.power(population.chromosomes_fitness,x) / np.sum(np.power(population.chromosomes_fitness,x))
        if np.sum(population.chromosomes_fitness) <0:
            offset = min(population.chromosomes_fitness)
            positive_fit = [fit - offset + 1 for fit in population.chromosomes_fitness]
            select_probs =  np.power(positive_fit,x) / np.sum(np.power(positive_fit,x))
            
      
        print('crossing-over... p=', population.crossover_prob)
        offsprings = []
        jobs=[]
        dk = int(initial_n_chr/2)
        random_seeds=[np.random.randint(2**32 - 1) for i in range(dk)]
        population.chromosomes= np.array(population.chromosomes)
        # if ctr>=1: # do tournament for granting population diversity
        # parents = [population.tournament_selection(2, select_probs) for _ in range(dk)]
        # else: # normally don't
        parents = [population.chromosomes[np.random.choice(range(elites_len), 2, replace=False, p=select_probs)] 
                    for _ in range(dk)]
        for i,parent in enumerate(parents):
            jobs.append(pool.apply_async(population.crossover, [parent[0], parent[1], random_seeds[i]]))
        for j in jobs:
            child1,child2, child3, child4=j.get()
            offsprings.append(child1)
            offsprings.append(child2)
            if child3!=None:
                offsprings.append(child3)
                offsprings.append(child4)
        #------------------------------#

        #----------------MUTATION----------------#
        print('mutating... p=', population.mutation_prob)    
        mutated_offsprings = [population.mutate(child, generation//2) for child in offsprings]    
           

        #------------------------------#

        #-----------NEXT GENERATION-----------# 
        # population = elite
        # mutated_offsprings += [population.best_individual,]
        population = Population(mutation_prob=population.mutation_prob-(population.mutation_prob/n_generations), crossover_prob=population.crossover_prob, max_elite=population.max_elite, environment=environment)
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
        sid=1395713694#1919494547#1438423823#2468609729 #123456          #2245427923
    if sid=='r':
        sid=np.random.randint(2**32 - 1)
        print('using ', sid)
    else:
        sid=int(sid)

    abs_time_start = time.time()

    # environment = Environment(
    #         env_id          = 'CartPole-v0',
    #         n_episodes      = 100,
    #         bins            = (7, 4, 7, 6) 
    #     )
    # population = Population(
    #     mutation_prob   = 0.9,
    #     crossover_prob  = 0.9,
    #     max_elite       = 12,
    #     environment     = environment
    # )
    # all_populations = evolve(
    #     population, 
    #     environment, 
    #     initial_n_chr = 185, 
    #     n_generations = 5,
    #     seed          = sid,
    #     genotype_len  = 22,
    #     MAX_DEPTH     = 5,
    #     MAX_WRAP=3
    # )


    # environment = Environment( 
    #         env_id          = 'MountainCar-v0', # 1. prova con seed diversi
    #         n_episodes      = 100,
    #         bins            = (6,5),#np.full(128, 6, int)#(10,10) # 2. ho provato 9,10 e 10,9 ma danno meno di 116 (CON SEED 1234 !!!!!! INSERISCILO A MANO)
    #     )                               #prova pong cambiando gli elementi 8 10 11 12 15- 21 50 51 52 55 57- 59- 61 122 123 e gli altri lasciali di base a 2
    # population = Population(
    #     mutation_prob   = 0.95,
    #     crossover_prob  = 0.6,
    #     max_elite       = 50, # WAS 50 3. 27 no. 26 (115),  25 E 23 mi ha dato -116.85 (CHR 387 GEN 3, then not converged enymore) -> prova a diminuire max_elite
    #     environment     = environment
    # )
    # all_populations = evolve(
    #     population, 
    #     environment, 
    #     initial_n_chr = 150, # WAS 250 4.  250 sì(116), 300 no (115)!!!
    #     n_generations = 25, #generation 5 -> 115
    #     seed          = sid,
    #     genotype_len  = 100, # 5. questi 
    #     MAX_DEPTH     = 6, # 5.poi cambia questi lasciando tutto invariato
    #     MAX_WRAP      = 12 # 5.questi
    # )


    environment = Environment(
            env_id          = 'Acrobot-v1',
            n_episodes      = 100,
            bins            = (13,13,13,13,13,13) #gen 9 -80.61
        )
    population = Population(
        mutation_prob   = 0.9,
        crossover_prob  = 0.8,
        max_elite       = 80,
        environment     = environment
    )
    all_populations = evolve(
        population, 
        environment, 
        initial_n_chr = 200, 
        n_generations = 10,
        seed          = sid,
        genotype_len  = 150, #aymenta questo (meno no)
        MAX_DEPTH     = 6,
        MAX_WRAP=10
    )

    # environment = Environment(
    #         env_id          = 'LunarLander-v2',
    #         n_episodes      = 100,
    #         bins            = (5,5,6,7,6,7,1,1) #lander_pos_x, lander_pos_y, lander_vel_x, lander_vel_y, lander_ang, _lander_vel_ang, _2 leg_contact (0 or 1)
    #     )
    # population = Population(
    #     mutation_prob   = 0.9,
    #     crossover_prob  = 0.9,
    #     max_elite       = 30,
    #     environment     = environment
    # )
    # all_populations = evolve(
    #     population, 
    #     environment, 
    #     initial_n_chr = 150, 
    #     n_generations = 30,
    #     seed          = sid,
    #     genotype_len  = 120,
    #     MAX_DEPTH     = 8,
    #     MAX_WRAP=18
    # )


    abs_time= time.time() - abs_time_start
    
    #---------------plotting-------------#
    print('Plotting ... ')


    for generation, population in enumerate(all_populations):
        population.best_individual.tree_to_png(generation)
        population.best_individual.generate_solution(generation, to_file=True)

    ep_len = len(all_populations[0].chromosomes_scores[0])
    z_axys = np.arange(ep_len)
    plt.rc('xtick', labelsize=20)     
    plt.rc('ytick', labelsize=20)
    for i,population in enumerate(all_populations):
        ax= plt.figure(figsize=(20, 19)).add_subplot(111, projection='3d')
        best_idx = np.argmax(population.chromosomes_fitness)
        if len(population.chromosomes_scores)>12:
            low =  0 if best_idx-5<0 else best_idx-10 if best_idx+5>=len(population.chromosomes_scores) else best_idx-5
            high = len(population.chromosomes_scores) if best_idx+5>=len(population.chromosomes_scores)-1 else best_idx+5
            scores = np.array(population.chromosomes_scores)[range(low, high)] 
        else:
            scores = population.chromosomes_scores
        ax.set_xticks( np.arange(len(scores)))
        
        for j,score in enumerate(scores):
            ax.plot(np.full(ep_len, j, int)  , z_axys, score, zorder=j)

        ax.set_zlabel("Rewards", fontsize=27, labelpad=20)
        ax.set_ylabel("Episode", fontsize=27, labelpad=20)
        ax.set_xlabel("Chromosome", fontsize=27, labelpad=20)
        
        title=  environment.env.spec.id+" solved in {} generations\n".format(len(all_populations)-1)
        title += "time elapsed = {} sec\n".format(abs_time)
        title += "GENERATION [ {} / {} ]".format(i, len(all_populations)-1)
        plt.title(title, fontsize=30)
        save_dir = './outputs/GEN-{}/'.format(i)
        plt.savefig(save_dir+'plot.png', bbox_inches='tight')
    print('used seed = ', sid)
    #--------------evaluate--------------------#
    wrap = 'y'#input('Do you want to run the evolved policy and save it?    [y/N]    ')
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
    
