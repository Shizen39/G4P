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


#TO ASK : MAX_DEPTH and MAX_WRAP are chromosom-specific? (e.g. can I have different max_depth and max_wrap from two different chromosome?)
#         maybe not couse of multiprocessing? (they are global variables in Grammatica_Evolution_mapper.py)
#

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
        chromosome_score = deque(maxlen = self.env.spec.trials)
        for episode in range(self.n_episodes):
            reward = self.run_one_episode(chromosome, episode, prnt, render)
            chromosome_score.append(reward)
            if np.mean(chromosome_score) >= self.env.spec.reward_threshold and episode>=self.env.spec.trials: #getting reward of 195.0 over 100 consecutive trials
                break 
        print("Chromosome ",i,"fitness = ",np.mean(chromosome_score))
        return list(chromosome_score)
    
    def parallel_evaluate_population(self, pool, population):
        population_scores = [] 
        jobs=[]
        for i,chromosome in enumerate(population):                                           #population_scores = Parallel(n_jobs=-1)(delayed(evaluate_policy)(env, chromosome, n_episodes) for chromosome in population if not converged)
            jobs.append(pool.apply_async(self.evaluate_chromosome, [chromosome, i]))
        for j in jobs:
            if not self.converged:
                if not j.ready():
                    j.wait()
                score=j.get()
                population_scores.append(score)
                if np.mean(score)>=self.env.spec.reward_threshold:
                    self.converged = True
            else:
                pool.terminate()
        return population_scores



#-----prepare gym-----#
env = Environment(env_id='CartPole-v0', n_episodes=n_episodes, bins=(3,3,3,3))



#---------INIT POPULATION----------#
#use ramped for deriv. tree to ensure the validity of initial pop (e.g. all leaf are terminals)



#---------EVAL POPULATION----------#

#return best_chromosome_tree

#---------CROSSING OVER---------#
#anytree stuffs

#---------MUTATE----------#
#anytree stuffs

#---------NEXT GENERATION----------#




#---------RUN BEST CHROMOSOME---------#
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