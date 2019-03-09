from anytree import Node, RenderTree
from anytree.exporter import DotExporter
from anytree.dotexport import RenderTreeGraph
from anytree import PreOrderIter
import numpy as np

import Grammatical_Evolution_mapper as GE

def generate_tree_from_int(int_genotype, method, MAX_DEPTH, MAX_WRAP, to_png=False, to_shell=False):
    '''
    Generate (phenotype) derivation tree from a (genotype) list of int
    Genotype-phenotype mapping function is the MOD operator between genes of the genotype and rules of the Grammar. 
    Parameters : int_genotype (list of integer genes)
                 MAX_DEPTH (maximum depth of the generated derivation tree)
                 MAX_WRAP (maximum number of time that wrapping operator is applied to int_genotype)
    Return value : tree_phenotype (root of the generated tree)
    '''
    root = Node('('+str(0)+')expr-start', label='expr', code='')                      # root of derivation tree
    tree_pheontype = GE.start_derivating(int_genotype, root, method, MAX_DEPTH, MAX_WRAP, _initial_gene_seq = int_genotype)

    if to_shell:
        for pre, _, node in RenderTree(tree_pheontype):                                # print tree on terminal
            print("{}{}".format(pre, node.name)) 
    if to_png:
        RenderTreeGraph(tree_pheontype, nodeattrfunc=lambda node: 'label="{}"'.format( # export tree .png file
            node.label)).to_picture("tree_pheontype.png")                              #
    return tree_pheontype


def generate_program_from_tree(tree_pheontype, write_to_file=False):
    '''
    Generate (solution) chromosom with program representation obtained doing
    PRE-ORDER starting from (phenotype) argument passed node (root) and collecting
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