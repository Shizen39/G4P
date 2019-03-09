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

import Genetic_Programming as GP


#TO ASK : MAX_DEPTH is chromosom-specific? (e.g. can I have different max_depth from two different chromosome?)
#         
#

class Evolution():
    def __init__(self, n_chromosomes, n_generations, mutation_prob, max_elite, seed, genotype_len, MAX_DEPTH):
        self.genotype_len = genotype_len
        self.MAX_DEPTH = MAX_DEPTH

        self.n_chromosomes = n_chromosomes
        self.n_generation = n_generations
        self.mutation_prob = mutation_prob
        self.max_elite = max_elite
        self.seed = seed
    
    def create_random_population(self):
        min_genotype_len = self.genotype_len - int(self.genotype_len/2)
        max_genotype_len = self.genotype_len + int(self.genotype_len/2)
        genotype_population = [Chromosome(np.random.randint(min_genotype_len, max_genotype_len)) for _ in self.n_chromosomes]
        population = [chromosome.generate_phenotype('grow', self.MAX_DEPTH) for chromosome in genotype_population[:int(len(genotype_population)/2)]]
        population += [chromosome.generate_phenotype('full', self.MAX_DEPTH) for chromosome in genotype_population[int(len(genotype_population)/2):]]
        return population

    def select_elite(self, population, population_fitness, elite_threashold, population_scores):
        elite = [e for i, e in enumerate(population)            # survive only those fitness 
                if population_fitness[i] >= elite_threashold]   # is greater then  mean of all fitness
        elite_score = [e for i, e in enumerate(population_scores) 
                if population_fitness[i] >= elite_threashold]

        elite_fitness = list(np.mean(elite_score, axis=1))

        if len(elite) > self.max_elite:
            while len(elite)>self.max_elite:
                rm= np.argmin(elite_fitness)
                elite.pop(rm)
                elite_score.pop(rm)
                elite_fitness.pop(rm)
        return elite, elite_fitness

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
                mutated[k] = np.random.randint(0,env.action_space.n)
        return mutated


class Chromosome():
    def __init__(self, GENOTYPE_LEN):
        ''' 
        Parameters : GENOTYPE_LEN (number of genes of the genotype)
        Attributes : - genotype (list of integer that corresponds to the set of genes of the genotype)
                     - phenotype (derivation tree rappresentation of the chromosome, that corresponds to the set of genes (nodes) encoded by the genotype)
                     - solution (python code rappresentation of the chromosome, that corresponds to the set of genes (line of codes) translated by the phenotype)
        '''
        self.genotype = [np.random.randint(1,3)]+list(np.random.randint(0,1000,size=GENOTYPE_LEN-1))
        self.phenotype = None
        self.solution = None
        self.fitness = None

    def generate_phenotype(self, method, MAX_DEPTH, MAX_WRAP=5, to_png=False, to_shell=False):
        '''
        Generate a tree from the self.genotype, and assign it at self.phenotype
        Parameters : MAX_DEPTH (maximum depth of the generated phenotypes' derivation trees)
                     MAX_WRAP  (maximum number of time that wrapping operator is applied to genotype)
        
        '''
        self.phenotype = GP.generate_tree_from_int(self.genotype, method, MAX_DEPTH, MAX_WRAP, to_png=to_png, to_shell=to_shell)
    
    def generate_solution(self, write_to_file=False):
        '''
        Generate a python program from the self.phenotype, assigning it at self.solution
        '''
        self.solution = GP.generate_program_from_tree(self.phenotype, write_to_file= write_to_file)

    def execute_solution(self, observation, states):
        '''
        Execute generated python program
        Parameters : observation (list of states of the environment)
                     states (list of all possible states of an observation of the environment)
        Return value: an action
        '''
        return GP.get_action_from_program(observation, states, self.solution)


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