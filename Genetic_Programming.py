from anytree import Node, RenderTree
from anytree.exporter import DotExporter
from anytree.dotexport import RenderTreeGraph
from anytree import PreOrderIter
import numpy as np

import Grammatical_Evolution_mapper as GE

def generate_tree_from_int(int_genotype, MAX_DEPTH, MAX_WRAP, export_to_png=False, print_to_shell=False):
    '''
    Generate derivation tree from int_genotype
    using MOD genes of the int_genotype as genotype-phenotype mapping function. 
    Parameters : int_genotype (list of integer genes)
                 MAX_DEPTH (maximum depth of the generated derivation tree)
                 MAX_WRAP (maximum number of time that wrapping operator is applied to int_genotype)
    Return value : tree_phenotype (root of the generated tree)
    '''
    GE.MAX_DEPTH = MAX_DEPTH   # max depth of the tree
    GE.MAX_WRAP = MAX_WRAP # max number of time wrapping operator is applied
            
    root = Node('('+str(0)+')expr-start', label='expr', code='')                      # root of derivation tree
    tree_pheontype = GE.generate_derivation_tree(int_genotype, root)

    if print_to_shell:
        for pre, _, node in RenderTree(tree_pheontype):                                # print tree on terminal
            print("{}{}".format(pre, node.name)) 
    if export_to_png:
        RenderTreeGraph(tree_pheontype, nodeattrfunc=lambda node: 'label="{}"'.format( # export tree .png file
            node.label)).to_picture("tree_pheontype.png")                              #
    return tree_pheontype


def generate_program_from_tree(tree_pheontype, write_to_file=False):
    '''
    Generate chromosom with program representation obtained doing
    PRE-ORDER starting from argument passed node (root) and collecting
    all node.code properties, concatenating them in a variable.
    Return value: variable containing programs' code
    '''
    program_chromosome="def get_action(observation, states):\n\t"                   # Prepare program whit func def and return value
    for node in PreOrderIter(tree_pheontype):   
        program_chromosome+= node.code                                              # get generated program
    program_chromosome+="\n\treturn action"                                          #

    if write_to_file:
        file = open('program_chromosome.py', 'w')                                       # Create file and write in generated programs'string
        file.write(program_chromosome)                                                  #
        file.close()   
    return program_chromosome
#------------------------------------------#


def subdivide_observation_states(env, bins):
    '''
    Subdivide each continous state (i) of an observation in 
    bins[i] number of discrete states (e.g. states[state] = [low_bound, ... i, ..., high_bound]);
    Return value: list of states
    '''
    sp_low, sp_up = env.observation_space.low, env.observation_space.high
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


def get_action_from_program(observation, states, program_chromosome):
    '''
    Run program_chromsome as python code that return an action.
    program_chromosome could be a string of python code or 
    a python file; in the latter case pass program_chromosome 
    parameter as open("./file.py").read()
    Return value: an action
    '''
    loc={}
    exec(program_chromosome, {}, loc)
    try:
        action=loc['get_action'](observation, states)
    except UnboundLocalError:   #observation did not pass through any if else
        print('Assign low fitness')
        action=0
    
    return action