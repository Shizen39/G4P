import numpy as np
import gym.wrappers
import gym.spaces
import gym

import ggp
import Grammar

CHROMOSOME_LEN = 10     # number of genes in a chromosomes
Grammar.MAX_DEPTH = 5   # max depth of the tree
Grammar.MAX_RESTART = 5 # max number of time wrapping operator is applied


#---prepare gym?-----#
env = gym.make('CartPole-v0')
states = ggp.subdivide_observation_states(env, bins = (3, 3, 3, 3))



#---------INIT POPULATION----------#
int_chromosome = [1]+list(np.random.randint(0,1000,size=CHROMOSOME_LEN-1))

tree_chromosome = ggp.generate_tree_from_int(int_chromosome, True, False)

#---------EVAL POPULATION----------#
program_chromosome= ggp.generate_program_from_tree(tree_chromosome)
#return best_chromosome_tree

#---------CROSSING OVER---------#
#anytree stuffs

#---------MUTATE----------#
#anytree stuffs

#---------NEXT GENERATION----------#




#---------RUN BEST CHROMOSOME---------#
ggp.generate_program_from_tree(tree_chromosome, write_to_file=True)
env.seed(0)

obs = env.reset()
for timestep in range(100): #loop dei timesteps
    env.render()
    action = ggp.get_action_from_program(obs, states, open('./program_chromosome.py').read())
    obs, reward, done, info = env.step(action)
env.close()

#TODO: LOOK AT GRAMMARS' if i_gene_same >= len(gene_seq): 
#       COUSE IT RETURNS SOMETHING BUT IN CASE I WANT TO 
#       INSERT A LEAF NODE INSTEAD