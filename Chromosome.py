'''
This file define the Chromosome representation using the grammar defined in 
Grammatical_Evolution_mapper.py

In particular it defines the representations of a single chromosome as a:
genotype (a sequence of random integer genes)
phenotype (a derivation tree builded using both genotype and Grammar rules)
solution (a python code generated through the phenoype)

and defines the corresponding functions to generate them.
'''


from anytree import Node, RenderTree
from anytree.exporter import DotExporter
from anytree.dotexport import RenderTreeGraph
from anytree import PreOrderIter
import numpy as np

import Grammatical_Evolution_mapper as GE


class Chromosome():
    ''' 
    Chromosome defines the representations of a single individuals as:
    its genotype (sequence of int), phenotype (tree) and solution (string code)
    
    Args:
        GENOTYPE_LEN (int): number of genes of the genotype
    
    Attributes: 
        genotype (list(int)): the set of genes of the genotype
        phenotype (AnyTree.Node): derivation tree rappresentation of the chromosome, that corresponds to the set of genes (nodes) encoded by the genotype
        solution (str): python code rappresentation of the chromosome, that corresponds to the set of genes (line of codes) translated by the phenotype
    '''
    def __init__(self, GENOTYPE_LEN):
        self.genotype = [np.random.randint(1,3)]+list(np.random.randint(0,1000,size=GENOTYPE_LEN-1)) # ensure that it starts with rule 1 or 2
        self.phenotype = None
        self.solution = None

    def generate_phenotype(self, method, MAX_DEPTH, MAX_WRAP, to_png=False, to_shell=False):
        '''
        Generate phenotype from genotype (derivation tree from a list of int).
        Genotype-phenotype mapping function is the MOD operator between genes of the genotype and rules of the Grammar. 

        Args:
            MAX_DEPTH (int): maximum depth of the generated phenotypes' derivation trees
            MAX_WRAP  (int): maximum number of time that wrapping operator is applied to genotype
            to_png (boolean): export tree on png file
            to_shell (boolean): print tree on shell
        '''
        root = Node('('+str(0)+')expr-start', label='expr', code='')                      # root of derivation tree
        self.phenotype = GE.start_derivating(self.genotype, root, method, MAX_DEPTH, MAX_WRAP)
        if to_shell:
            for pre, _, node in RenderTree(self.phenotype):                                # print tree on terminal
                print("{}{}".format(pre, node.name)) 
        if to_png:
            RenderTreeGraph(self.phenotype, nodeattrfunc=lambda node: 'label="{}"'.format( # export tree .png file
                node.label)).to_picture("tree_phenotype_{}.png".format(self))                              #


    def generate_solution(self, to_file=False):
        '''
        Generate solution (python program)
        The program representation of the phenotype is obtained doing PRE-ORDER starting from root node (phenotype)
        and collecting all node.code properties, concatenating them in a string variable.

        Args:
            to_file (bool): write program to a file
        '''
        program_chromosome="def get_action(observation, states):\n\t"                   # Prepare program whit func def and return value
        for node in PreOrderIter(self.phenotype):   
            program_chromosome+= node.code                                              # get generated program
        program_chromosome+="\n\treturn action"                                          #
        
        self.solution = program_chromosome

        if to_file:
            file = open('program_solution_{}.py'.format(self), 'w')                                    # Create file and write in generated programs'string
            file.write(self.solution)                                                  #
            file.close()   

    def execute_solution(self, observation, states):
        '''
        Execute self.solution as python program
        
        Args:
            observation (list(float)): list of states of the environment
            states (list(list(float))): list of all possible states of an observation of the environment
        
        Returns: an action
        '''
        loc={}
        exec(self.solution, {}, loc)
        try:
            action=loc['get_action'](observation, states)
        except UnboundLocalError:   #observation did not pass through any if else
            #print('Assign low fitness')
            action= np.random.randint(0,2) #there (action_space.n)
        
        return action