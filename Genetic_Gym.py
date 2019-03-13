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

from anytree import Node, PreOrderIter, RenderTree, LevelOrderGroupIter
from anytree.dotexport import RenderTreeGraph
from anytree.exporter import DotExporter
from anytree.search import find


from collections import deque
import matplotlib.pyplot as plt         
from mpl_toolkits.mplot3d import Axes3D
from multiprocessing import Pool
import multiprocessing
#from joblib import Parallel, delayed
import copy

from Chromosome import Chromosome
from Grammatical_Evolution_mapper import Parser



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
        print("Survived [ ",len(elites)," / ",len(self.chromosomes)," ] chromosomes")
        self.chromosomes          = elites
        self.chromosomes_scores   = elite_scores
        self.chromosomes_fitness  = elite_fitness

    def crossover(self, parent_A, parent_B, a, b, reward_threshold): #TODO: PASS BORDER ON START_DER
        '''  
        Produce offsprings combining random parts of two chromosomes and generating two offsprings
        parent1, parent2
        child1 = parent1 + swap_random_subtree(parent2)
        child2 = parent2 + swap_random_subtree(parent1)
        '''
        child_A = copy.deepcopy(parent_A)
        child_B = copy.deepcopy(parent_B)
        tree_a = copy.deepcopy(parent_A.phenotype)
        tree_b = copy.deepcopy(parent_B.phenotype)

        
        if tree_a.children[0].label == tree_b.children[0].label:
            # if both nodes tree have the same first label (both or cond or expr)
            # choose random node from first expr or second
            if tree_a.children[0].label == 'cond':
                name = np.random.choice(['expr_i', 'expr_e'])
            else:
                name = np.random.choice(['expr_a', 'expr_b'])
            selected_node_A = [child for child in tree_a.children if child.name.rsplit(')')[1].rsplit('_mut')[0] == name][0]
            selected_node_B = [child for child in tree_b.children if child.name.rsplit(')')[1].rsplit('_mut')[0] == name][0]
        # elif tree_a.children[0].label=='expr' and tree_b.children[0].label=='expr':
        else:
            # else one tree is cond-expr_a-expr_b and the other expr_a-expr_ (they have different code!)
            
            # scendo di livello finchè in quel livello ho due label uguali
            # levels_A = []
            # for children in LevelOrderGroupIter(tree_a):
            #     level = [node for node in children if node.label=='expr'or node.label=='cond']
            #     if level!=[]:
            #         levels_A.append(level)
            # levels_B = []
            # for children in LevelOrderGroupIter(tree_b):
            #     level = [node for node in children if node.label=='expr'or node.label=='cond']
            #     if level!=[]:
            #         levels_B.append(level)
            
            # lvl=2 # skip first two layers
            # selected_node_B=[]
            # while lvl<len(levels_A) and lvl<len(levels_B):
            #     print(lvl)
            #     selected_node_A = np.random.choice(levels_A[lvl])
            #     if selected_node_A.label=='cond':
            #         selected_node_A = selected_node_A.parent.children[1] if len(selected_node_A.parent.children)==2 else selected_node_A.parent.children[np.random.randint(1,3)]# expr_i or expr_e
            #         levels_B[lvl] = [node for node in levels_B[lvl] if node.label == 'cond'] # reduce to cond only
            #         levels_B[lvl] = [node for node in levels_B[lvl] if node.name.rsplit(')')[1].rsplit('_mut')[0] == selected_node_A.name.rsplit(')')[1].rsplit('_mut')[0]] # reduce to expr_i or expr_e only
            #         if levels_B[lvl] == []:
            #             lvl+=1
            #             continue
            #         selected_node_B = np.random.choice(levels_B[lvl])
            #         break

            #     else: #==expr
            #         if selected_node_A.name.rsplit(')')[1].rsplit('_mut')[0] in ['expr_i', 'expr_e']:
            #             levels_B[lvl] = [node for node in levels_B[lvl] if node.label == 'cond'] # reduce to cond only
            #             levels_B[lvl] = [node for node in levels_B[lvl] if node.name.rsplit(')')[1].rsplit('_mut')[0] == selected_node_A.name.rsplit(')')[1].rsplit('_mut')[0]] # reduce to expr_i or expr_e only
            #             if levels_B[lvl] == []:
            #                 lvl+=1
            #                 continue
            #             selected_node_B = np.random.choice(levels_B[lvl])
            #             break

            #         else:
            #             levels_B[lvl] = [node for node in levels_B[lvl] if node.label == 'expr'] # reduce to expr only
            #             levels_B[lvl] = [node for node in levels_B[lvl] if node.name.rsplit(')')[1].rsplit('_mut')[0] == selected_node_A.name.rsplit(')')[1].rsplit('_mut')[0]] # reduce to expr_a or expr_b only
            #             if levels_B[lvl] == []:
            #                 lvl+=1
            #                 continue
            #             selected_node_B = np.random.choice(levels_B[lvl])
            #             break

        # if selected_node_B == []:
            return parent_A, parent_B


        

        #---------------------------------------#
        color = '/blues9/1'
        # orange
        if selected_node_A.color.rsplit('/',1)[0]=='/oranges9':
            border = '/oranges9/9'
            color = '/blues9/1'
            # coloring all its childs the same
            for child in PreOrderIter(selected_node_A):
                child.color=color
                child.border=border
        # gray or blue
        else: 
            for child in PreOrderIter(selected_node_A):
                if child.color.rsplit('/',1)[0]=='/oranges9':
                    border = '/oranges9/9'
                    color = '/blues9/1'
                else:
                    border = '/blues9/9'
                    if int(child.color.rsplit('/',1)[1])<9:
                        color = '/blues9/'+str(int(child.color.rsplit('/',1)[1])+1)
                    else:
                        color = '/blues9/1'

                child.color=color
                child.border=border
            border = '/blues9/9'
            if int(selected_node_A.color.rsplit('/',1)[1])<9:
                color = '/blues9/'+str(int(selected_node_A.color.rsplit('/',1)[1])+1)
            else:
                color = '/blues9/1'
        selected_node_A.color=color
        selected_node_A.border=border

        #--------------------------------------#
        
        
        color = '/blues9/1'
        # orange
        if selected_node_B.color.rsplit('/',1)[0]=='/oranges9':
            border = '/oranges9/9'
            color = '/blues9/1'
            # coloring all its childs the same
            for child in selected_node_B.children:
                child.color=color
                child.border=border
        # gray or blue
        else: 
            for child in selected_node_B.children:
                if child.color.rsplit('/',1)[0]=='/oranges9':
                    border = '/oranges9/9'
                    color = '/blues9/1'
                else:
                    border = '/blues9/9'
                    if int(child.color.rsplit('/',1)[1])<9:
                        color = '/blues9/'+str(int(child.color.rsplit('/',1)[1])+1)
                    else:
                        color = '/blues9/1'

                child.color=color
                child.border=border
            border = '/blues9/9'
            if int(selected_node_B.color.rsplit('/',1)[1])<9:
                color = '/blues9/'+str(int(selected_node_B.color.rsplit('/',1)[1])+1)
            else:
                color = '/blues9/1'
        selected_node_B.color=color
        selected_node_B.border=border
        #-----------------------------------------#
        print('Crossingover... NODE', selected_node_A.name, selected_node_B.name)
        
        tmp_B = copy.deepcopy(selected_node_B)
        tmp_B.parent=None
        tmp_A = copy.deepcopy(selected_node_A)
        # modify the list of parents' selected_node childrens
        siblings_A = list(selected_node_A.parent.children)
        # sobstituting it with mutated one
        siblings_A[selected_node_A.parent.children.index(selected_node_A)] = tmp_B
        
        # and reassigning it
        selected_node_A.parent.children = tuple(siblings_A)
        # set mutated chromosomes' phenotype as mutated root        
        child_A.phenotype = tree_a
        #-----------
        #selected_node_B.parent = tree_b
        # modify the list of parents' selected_node childrens
        siblings_B = list(selected_node_B.parent.children)
        # sobstituting it with mutated one
        siblings_B[selected_node_B.parent.children.index(selected_node_B)] = tmp_A
        # and reassigning it
        tmp_A.parent = selected_node_B.parent
        selected_node_B.parent.children = tuple(siblings_B)
        # set mutated chromosomes' phenotype as mutated root  
        child_B.phenotype = tree_b

        child_A.tree_to_png(0)
        child_B.generate_solution(0, to_file=True)
        child_B.tree_to_png(0)
        child_A.generate_solution(0, to_file=True)
        

        
        return child_A, child_B

    def mutate(self, chromosome, p=0.05): #TODO: MODIFY BORDER='9' AS ON CROSSOVER (CREATE VAR BORDER IN IF COLOR)
        '''
        Mutate genes of a chromosomes.
        retrieve tree max_depth (height?)
        arange an array of levels of crescent probability (one prob for each level)
        random choose a level
        random choose a node inside that level (or rand chose an id?)
        do above things
        '''
        root = chromosome.phenotype
        # Iterate over tree using level-order strategy returning lists of nodes for every level (e.g. levels[level][node])
        levels = [[node for node in children if node.label=='expr'or node.label=='cond'] 
                    for children in LevelOrderGroupIter(root)]
        while levels[-1]==[]:
            levels.pop()
        max_depth = len(levels)
        # random select a level in which a node will be random selected for mutation
        # the higher the level - the higher the probability is to be choosed
        levels_prob = np.arange(max_depth) / np.sum(np.arange(max_depth))
        level = np.random.choice(levels, p=levels_prob)

        # random choose a node in that level and retrieve its id
        selected_node = np.random.choice(level)
        level_number=len(level)
        mut_node_id = int(''.join(filter(str.isdigit, selected_node.name.rsplit('_mut')[0])))
        color = '/oranges9/1'
        if selected_node.color.rsplit('/',1)[0]=='/blues9':
            border = '/blues9/9'
            color = '/oranges9/1'
        else: 
            border = '/oranges9/9'
            if int(selected_node.color.rsplit('/',1)[1])<9:
                color = '/oranges9/'+str(int(selected_node.color.rsplit('/',1)[1])+1)
            else:
                color = '/oranges9/1'
        print("Mutating... NODE ",selected_node.name)
        if selected_node.label == 'expr':
            # create new rando genotype of i_gen + n_descendents lenght
            mut_genotype = [np.random.randint(1,3)]+list(np.random.randint(0,1000,size=mut_node_id + len(selected_node.descendants)))
            # create new mutated node (root)
            mutated = Node(selected_node.name+'_mut_'+color.rsplit('/',1)[1], label='expr', code=selected_node.code, indent=selected_node.indent, color=color, border=border)
            # instantiate a new parser 
            parser = Parser(mut_genotype, mutated, 'full', MAX_DEPTH=max_depth-level_number, MAX_WRAP=max_depth)
            # set parser parameters to those of the selected_node and start parsing
            parser.i_gene = mut_node_id+1
            if selected_node.name.rsplit(')')[1].rsplit('_mut')[0] in ['expr_a', 'expr_b']:
                indent = selected_node.indent
            else:
                indent = selected_node.indent+1
            mutated = parser.start_derivating('expr', tree_depth=level_number, indent=indent, extra_id='_mut_'+color.rsplit('/',1)[1])
        
        elif selected_node.label == 'cond':
            mut_genotype = list(np.random.randint(0,1000,size=mut_node_id + len(selected_node.descendants)))
            # create new mutated node (root)
            mutated = Node(selected_node.name+'_mut_'+color.rsplit('/',1)[1], label='cond', code=selected_node.code, color=color, border=border)
            # instantiate a new parser 
            parser = Parser(mut_genotype, mutated, 'full', MAX_DEPTH=max_depth-level_number, MAX_WRAP=max_depth)
            # set parser parameters to those of the selected_node and start parsing
            parser.i_gene = mut_node_id+1
            mutated = parser.start_derivating('cond', tree_depth=level_number, extra_id='_mut_'+color.rsplit('/',1)[1])

        # modify the list of parents' selected_node childrens
        new_children = list(selected_node.parent.children)
        # sobstituting it with mutated one
        new_children[selected_node.parent.children.index(selected_node)] = mutated
        # and reassigning it
        selected_node.parent.children = tuple(new_children)
        # set mutated chromosomes' phenotype as mutated root
        chromosome.phenotype = root
        return chromosome




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
