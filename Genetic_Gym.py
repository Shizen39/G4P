'''
This file define the Genetic Operators using Chromosome.py representation of an individual and Gyms' specs.

In particular it defines two classes:
- Population: class that contains:
    - all Grammar Guided Genetic Programming parameters (n_chromosomes, n_generation, genotype_len...)
    - population attributes that defines population as a set of Chromosome objects
    - all Genetic Operators (initialize, natural_selection, crossingover, mutation)

- Environment: class that contains all gyms' specific functions in relation with the chromosome representation 
    
'''


import numpy as np
import gym.wrappers
import gym.spaces
import gym

import time
import multiprocessing


from collections import deque
import matplotlib.pyplot as plt         
from mpl_toolkits.mplot3d import Axes3D
from multiprocessing import Pool
import multiprocessing
#from joblib import Parallel, delayed

from Chromosome import Chromosome


#TO ASK : MAX_DEPTH is chromosom-specific? (e.g. can I have different max_depth from two different chromosome?)
#         
#

class Population():
    def __init__(self, n_chromosomes, mutation_prob, max_elite, seed):
        # Inizialization parameters
        self.n_chromosomes = n_chromosomes
        self.mutation_prob = mutation_prob
        self.max_elite = max_elite
        self.seed = seed
        # Population attribbutes
        self.chromosomes         = []
        self.chromosomes_scores  = []
        self.chromosomes_fitness = []
        self.survival_threashold = None
        self.best_individual     = None
    
    def initialize_chromosomes(self, MAX_DEPTH, genotype_len):
        min_genotype_len = genotype_len - int(genotype_len/2)
        max_genotype_len = genotype_len + int(genotype_len/2)
        # set genotype
        population = [Chromosome(GENOTYPE_LEN = np.random.randint(min_genotype_len, max_genotype_len)) for _ in self.n_chromosomes]
        # set phenotype
        for i, chromosome in enumerate(population):
            if i < int(len(population)/2):
                chromosome.generate_phenotype('grow', MAX_DEPTH)
            else:
                chromosome.generate_phenotype('full', MAX_DEPTH)
        return population

    def do_natural_selection(self):
        elites = [e for i, e in enumerate(self.population.individuals)                            # survive only those fitness 
                if self.population.individuals_fitness[i] >= self.population.survival_threashold]     # is greater then  mean of all fitness
        elite_scores = [e for i, e in enumerate(self.population.individuals_scores) 
                if self.population.individuals_fitness[i] >= self.population.survival_threashold]

        elite_fitness = list(np.mean(elite_scores, axis=1))

        if len(elites) > self.max_elite:
            while len(elites)>self.max_elite:
                rm= np.argmin(elite_fitness)
                elites.pop(rm)
                elite_scores.pop(rm)
                elite_fitness.pop(rm)
        self.chromosomes          = elites
        self.chromosomes_scores   = elite_scores
        self.chromosomes_fitness  = elite_fitness

    def crossover(self, parent_A, parent_B):
        '''  
        parent1, parent2
        child1 = parent1 + swap_random_subtree(parent2)
        child2 = parent2 + swap_random_subtree(parent1)
        '''
        pass

    def verify_crossover(self, child1, child2, offsprings):
        if child1 in offsprings:
            while child1 in offsprings:
                child1 = self.mutate(child1)
        if child2 in offsprings:
            while child2 in offsprings:
                child2 = self.mutate(child2)
        return child1, child2

    def mutate(self, child, p=0.05):
        '''
        child.phenotype = replace_random_subtree(child.phenotype)
        '''
        mutated = child
        for k in mutated:
            if np.random.uniform() < p:
                mutated[k] = np.random.randint(0,2)     # there (action_space.n)
        return mutated




class Environment():
    def __init__(self, env_id, n_episodes, bins):
        self.env = gym.make(env_id)
        self.n_episodes = n_episodes
        self.states =  self.subdivide_observation_states(bins)
        self.env_seed = self.env.seed(0)
        self.coverged = False
        self.pool = Pool()
        

    def subdivide_observation_states(self, bins):
        '''
        Subdivide each continous state (i) of an observation in 
        bins[i] number of discrete states (e.g. states[state] = [low_bound, ... i, ..., high_bound]);
        Return value: list of states
        '''
        sp_low, sp_up = self.env.observation_space.low, self.env.observation_space.high
        inf = np.finfo(np.float32).max
        div_inf = 7000**10
        bounds = []
        for i in range(len(sp_low)):
            if sp_low[i] == -np.inf:
                sp_low[i] = -inf
            if sp_up[i] == np.inf:
                sp_up[i] = inf
            bounds.append([sp_low[i]/div_inf if sp_low[i] == -inf else sp_low[i], 
                                        sp_up[i]/div_inf if sp_up[i] == inf else sp_up[i]])
        states = []
        for i, v in enumerate(bounds):
            x = np.histogram(v, bins[i])[1] # subdivide continous interval into equal spaced bins[i] intervals
            states.append(x)
        return states
    
    def run_one_episode(self, chromosome, episode, prnt=False, render=False):
        episode_reward = 0
        done = False
        obs = self.env.reset()
        while not done:
            if render: self.env.render()
            action = chromosome.execute_solution(obs, self.states)
            obs, reward, done, _ = self.env.step(action)
            episode_reward += reward
        if prnt: print('V' if episode_reward == 200 else 'X'," Ep. ",episode," terminated (", episode_reward, "rewards )")
        return episode_reward
    
    def evaluate_chromosome(self, chromosome, i, prnt=False, render=False):
        chromosome_scores = deque(maxlen = self.env.spec.trials)
        # set chromosome solutions' code
        chromosome.generate_solution()
        # run solution code
        for episode in range(self.n_episodes):
            reward = self.run_one_episode(chromosome, episode, prnt, render)
            chromosome_scores.append(reward)
            if np.mean(chromosome_scores) >= self.env.spec.reward_threshold and episode>=self.env.spec.trials: #getting reward of 195.0 over 100 consecutive trials
                break 
        print("Chromosome ",i,"fitness = ",np.mean(chromosome_scores))
        chromosome.fitness = np.mean(chromosome_scores)
        return list(chromosome_scores)
    
    def parallel_evaluate_population(self, population):
        population_scores = [] 
        jobs=[]
        for i,chromosome in enumerate(population):                                           #population_scores = Parallel(n_jobs=-1)(delayed(evaluate_policy)(env, chromosome, n_episodes) for chromosome in population if not converged)
            jobs.append(self.pool.apply_async(self.evaluate_chromosome, [chromosome, i]))
        for j in jobs:
            if not self.converged:
                if not j.ready():
                    j.wait()
                score=j.get()
                population_scores.append(score)
                if np.mean(score)>=self.env.spec.reward_threshold:
                    self.converged = True
            else:
                self.pool.terminate()
        return population_scores



#-----prepare gym-----#




#---------INIT POPULATION----------#
#use ramped for deriv. tree to ensure the validity of initial pop (e.g. all leaf are terminals)



#---------EVAL POPULATION----------#

#return best_chromosome_tree

#---------CROSSING OVER---------#
#anytree stuffs

#---------MUTATE----------#
#anytree stuffs

#---------NEXT GENERATION----------#




#---------  USAGE   --------#
env = Environment(env_id='CartPole-v0', n_episodes=n_episodes, bins=(3,3,3,3))

chromosome = Chromosome(GENOTYPE_LEN=10)
chromosome.generate_phenotype(method='full', MAX_DEPTH=6, MAX_WRAP=4, to_png=True)
chromosome.generate_solution(True)

env.env.seed(0)

obs = env.env.reset()
for timestep in range(100): #loop dei timesteps
    env.env.render()
    action = chromosome.execute_solution(obs, env.states)
    obs, reward, done, info = env.env.step(action)
env.env.close()


#TODO: 
#       Probably want to modify Grammatical_Ev as class so i can set global variables
#           as class variables and instantiate the class whit different variables 
#           (for parallel running, if i don't do this probably it braks couse of setting different values for the same variable in parallel)
