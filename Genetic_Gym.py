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
    '''
    This class represent a set of chromosomes that runs on a generation.
    Args:
        mutation_prob (float): probability of a single gene of a chromosome to mutate to a random gene
        max_elite (int): maximum number of chromosome that will survive after their evaluation (elites)

    Attributes:
        chromosomes (list(Chromosomes())): list of all chromosomes in that population
        chromosomes_scores (list(list(int))): list of all gym episodes rewards for each chromosomes
        chromosomes_fitness (list(float)): mean of chromosome_scores for each chromosomes scores
        survival_threashold (float): threashold that determine if a chromosome will survive or not (mean of all fitness values)
        best_indiviual (Chromosome()): best individual of that population (the one with highest fitness)
    '''
    def __init__(self, mutation_prob, max_elite):
        # Inizialization parameters
        self.mutation_prob = mutation_prob
        self.max_elite = max_elite
        # Population attribbutes
        self.chromosomes         = []
        self.chromosomes_scores  = []
        self.chromosomes_fitness = []
        self.survival_threashold = None
        self.best_individual     = None
    
    def initialize_chromosomes(self, n_chromosomes, genotype_len, MAX_DEPTH, MAX_WRAP=5, to_png=False):
        '''
        Initialize initial population (generation 0) by generating first a set of genotype
        and then - for each of them - generate the relative phenotype.

        Args:
            n_chromosomes (int): number of chromosomes in that population
            genotype_len: numbers of genes in the genotype. all genotypes would have random length 
                between genotype_len-(genotype_len/2) and genotype_len+(genotype_len/2)
            MAX_DEPTH (int): maximum depth of the generated phenotypes' derivation trees
            MAX_WRAP  (int): maximum number of time that wrapping operator is applied to genotype
            to_png (boolean): export each phenotype tree on png files
        Returns:
            population: a set of chromosomes with their .genotype and .phenotype already setted
        '''
        min_genotype_len = genotype_len - int(genotype_len/2)
        max_genotype_len = genotype_len + int(genotype_len/2)
        # set genotype
        population = [Chromosome(GENOTYPE_LEN = np.random.randint(min_genotype_len, max_genotype_len)) for _ in range(n_chromosomes)]
        # set phenotype
        for i, chromosome in enumerate(population):
            if i < int(len(population)/2):
                chromosome.generate_phenotype('grow', MAX_DEPTH, MAX_WRAP, to_png=to_png)
            else:
                chromosome.generate_phenotype('full', MAX_DEPTH, MAX_WRAP, to_png=to_png)
        self.chromosomes = population

    def do_natural_selection(self):
        '''
        Select the elites chromosomes from actual population (based on their fitness value)
        and remove non-selected chromosomes from the population.
        '''
        elites = [e for i, e in enumerate(self.chromosomes)                     # survive only those fitness 
                if self.chromosomes_fitness[i] >= self.survival_threashold]     # is greater then  mean of all fitness
        elite_scores = [e for i, e in enumerate(self.chromosomes_scores) 
                if self.chromosomes_fitness[i] >= self.survival_threashold]

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
        Produce offsprings combining random parts of two chromosomes and generating two offsprings
        parent1, parent2
        child1 = parent1 + swap_random_subtree(parent2)
        child2 = parent2 + swap_random_subtree(parent1)
        '''
        return parent_A, parent_B
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
        Mutate genes of a chromosomes
        '''
        mutated = child
        return mutated




class Environment():
    '''
    This class contains all gyms' specific functions in relation with the chromosome representation .
    Args:
        env_id (str): gym environment name
        n_episodes (int): number of episodes for each chromosome evaluation
        bins (list(int)): list that divide each state of all possible observations in discrete intervalls
    '''
    def __init__(self, env_id, n_episodes, bins):
        self.env = gym.make(env_id)
        self.n_episodes = n_episodes
        self.states =  self.subdivide_observation_states(bins)
        self.converged = False
        

    def subdivide_observation_states(self, bins):
        '''
        Use bins[i] to subdivide each continous state i of an observation in bins[i]
        number of discrete states (e.g. states[state] = [low_bound, ..., discrete_state_k, ..., high_bound]);
        Args:
            bins (list(int)): list that divide each state of all possible observations in discrete intervalls
        Returs: 
            states (list(list(float))): list of list of discrete states
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
        '''
        Run a single gym episode (composed by n timesteps), until that episode reach a terminal state (done = True).

        Args:
            chromosome (Chromosome()): actual chromosome that it's going to be evaluated
            episode (int): actual episode
        
        Returns:
            episode_reward (int): sum of all episode rewards (earned on each timesteps)
        '''
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
    
    def evaluate_chromosome(self, chromosome, i, to_file, prnt=False, render=False):
        '''
        Run self.n_episodes gym episodes with actual chromosome.
        
        Args: 
            chromosome (Chromosome())
        
        Returns:
            chromosome_scores (list(int)): list of all scores of the chromosome, of all episodes
        '''
        self.env.seed(0)
        chromosome_scores = deque(maxlen = self.env.spec.trials)
        # set chromosome solutions' code
        chromosome.generate_solution(to_file)
        # run solution code
        for episode in range(self.n_episodes):
            reward = self.run_one_episode(chromosome, episode, prnt, render)
            chromosome_scores.append(reward)
            if np.mean(chromosome_scores) >= self.env.spec.reward_threshold and episode>=self.env.spec.trials: #getting reward of 195.0 over 100 consecutive trials
                break 
        print("Chromosome ",i,"fitness = ",np.mean(chromosome_scores))
        return list(chromosome_scores)
    
    def parallel_evaluate_population(self, population, pool, to_file=False):
        '''
        Evaluate all chromosomes of the population (in parallel - using multiprocessing)

        Args:   
            population (list(Chromosome()))
            pool (multiprocessing.Pool)
            to_file (bool)
        
        Returns:
            population_scores (list(list(int))): list of all chromosomes list of rewards
        '''
        population_scores = [] 
        jobs=[]
        for i,chromosome in enumerate(population.chromosomes):                                           #population_scores = Parallel(n_jobs=-1)(delayed(evaluate_policy)(env, chromosome, n_episodes) for chromosome in population if not converged)
            jobs.append(pool.apply_async(self.evaluate_chromosome, [chromosome, i, to_file]))
        for j in jobs:
            if not self.converged:
                if not j.ready():
                    j.wait()    # ensure order
                score=j.get()
                population_scores.append(score)
                if np.mean(score)>=self.env.spec.reward_threshold:
                    self.converged = True
            else:
                pool.terminate()
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





#TODO: 
#       Probably want to modify Grammatical_Ev as class so i can set global variables
#           as class variables and instantiate the class whit different variables 
#           (for parallel running, if i don't do this probably it braks couse of setting different values for the same variable in parallel)
