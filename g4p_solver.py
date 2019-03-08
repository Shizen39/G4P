import numpy as np
import gym.wrappers
import gym.spaces
import gym

import time
import multiprocessing

import Genetic_Programming as GP


#TO ASK : MAX_DEPTH and MAX_WRAP are chromosom-specific? (e.g. can I have different max_depth and max_wrap from two different chromosome?)
#         maybe not couse of multiprocessing? (they are global variables in Grammatica_Evolution_mapper.py)
#

class Chromosome():
    def __init__(self, GENOTYPE_LEN):
        ''' 
        Parameters : GENOTYPE_LEN (number of genes of the genotype)
        Attributes : genotype (list of integer that corresponds to the set of genes of the genotype)
                     phenotype (derivation tree rappresentation of the chromosome, that corresponds to the set of genes (nodes) encoded by the genotype)
                     individual (python code rappresentation of the chromosome, that corresponds to the set of genes (line of codes) translated by the phenotype)
        '''
        self.genotype = [np.random.randint(0,4)]+list(np.random.randint(0,1000,size=GENOTYPE_LEN-1))
        self.phenotype = None
        self.individual = None

    def generate_phenotype(self, MAX_DEPTH=5, MAX_WRAP=5, export_to_png=False, print_to_shell=False):
        '''
        MAX_DEPTH : maximum depth of the generated phenotypes' derivation trees
        MAX_WRAP : maximum number of time that wrapping operator is applied to genotype
        '''
        self.phenotype = GP.generate_tree_from_int(self.genotype, MAX_DEPTH, MAX_WRAP, export_to_png=export_to_png, print_to_shell=print_to_shell)
    
    def generate_individual(self, write_to_file=False):
        self.individual = GP.generate_program_from_tree(self.phenotype, write_to_file= write_to_file)

#-----prepare gym-----#
env = gym.make('CartPole-v0')
states = GP.subdivide_observation_states(env, bins = (3, 3, 3, 3))



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
chromosome = Chromosome(10)
chromosome.generate_phenotype(6, 2, True, True)
chromosome.generate_individual(True)

env.seed(0)

obs = env.reset()
for timestep in range(100): #loop dei timesteps
    env.render()
    action = GP.get_action_from_program(obs, states, chromosome.individual)
    obs, reward, done, info = env.step(action)
env.close()
