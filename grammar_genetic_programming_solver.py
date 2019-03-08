import numpy as np
import gym.wrappers
import gym.spaces
import gym

import time
import multiprocessing

import ggp
import Grammar

#TO ASK : MAX_DEPTH and MAX_WRAP are chromosom-specific? (e.g. can I have different max_depth and max_wrap from two different chromosome?)

class Chromosome():
    def __init__(self, GENOTYPE_LEN):
        ''' 
        Parameters : GENOTYPE_LEN (number of genes of the genotype)
        Attributes : genotype (list of integer that corresponds to the set of genes of the genotype)
                     phenotype (derivation tree rappresentation of the chromosome, that corresponds to the set of genes (nodes) encoded by the genotype)
                     individual (python code rappresentation of the chromosome, that corresponds to the set of genes (line of codes) translated by the phenotype)
        '''
        self.genotype = [1]+list(np.random.randint(0,1000,size=GENOTYPE_LEN-1))
        self.phenotype = None
        self.individual = None

    def generate_phenotype(self, MAX_DEPTH, MAX_WRAP, export_to_png=False):
        '''
        MAX_DEPTH : maximum depth of the generated phenotypes' derivation trees
        MAX_WRAP : maximum number of time that wrapping operator is applied to genotype
        '''
        self.phenotype = ggp.generate_tree_from_int(self.genotype, MAX_DEPTH, MAX_WRAP, export_to_png=export_to_png)
    
    def generate_individual(self, write_to_file=False):
        self.individual = ggp.generate_program_from_tree(self.phenotype, write_to_file= write_to_file)

#-----prepare gym-----#
env = gym.make('CartPole-v0')
states = ggp.subdivide_observation_states(env, bins = (3, 3, 3, 3))



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
chromosome.generate_phenotype(5, 5, True)
chromosome.generate_individual(True)

env.seed(0)

obs = env.reset()
for timestep in range(100): #loop dei timesteps
    env.render()
    action = ggp.get_action_from_program(obs, states, chromosome.individual)
    obs, reward, done, info = env.step(action)
env.close()

#TODO: LOOK AT GRAMMARS' if i_gene_same >= len(gene_seq): 
#       COUSE IT RETURNS SOMETHING BUT IN CASE I WANT TO 
#       INSERT A LEAF NODE INSTEAD