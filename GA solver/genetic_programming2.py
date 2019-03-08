import gym.wrappers
import gym.spaces
import gym
from anytree import Node, RenderTree
from anytree.exporter import DotExporter
from anytree.dotexport import RenderTreeGraph
import numpy as np



CHROMOSOME_LEN = 10     # number of genes in a chromosomes

global MAX_RESTART      # max number of time that wrap func is applied to the sequence of genes
MAX_RESTART = 5

global MAX_DEPTH        # max depth of the tree
MAX_DEPTH = 5

global restart_ctr      # global counter that count number of time that wrap func is applied
restart_ctr=0

global i_gene            # global index that loops through all genes
i_gene = -1

#-----------DEF GRAMMAR AND TREE/PROGRAM GENERATOR-----------------#

#-----------RULES-----------------#
def wrap(gene_seq, ret):
    '''
    Function called when translation from genes to grammrs' rules runs out of genes
    '''
    global restart_ctr
    restart_ctr+=1
    if restart_ctr>MAX_RESTART:     #if wrapped too many times, return the same genes and a default terminal string
        return gene_seq, ret
    else:                           #if can wrap return 2*genes
        gene_seq += gene_seq                  
        return gene_seq, None
        

def expr(gene_seq, tree_depth, node):
    '''
    <expr>:   "if" <cond> ":" _NL _INDENT <expr> _NL                                            # 0
            | "if" <cond> ":" _NL _INDENT <expr> _NL "else:"_NL _INDENT <expr> _NL              # 1
            | "action =" ACTION                                                                 # 2
    '''
    global i_gene
    i_gene+=1
    if i_gene >= len(gene_seq):                 # if translation from genes to grammrs' rules runned out of genes (i_gene is the index of genes)
        gene_seq, ext = wrap(gene_seq,'pass')
        if ext!=None:
            return ext                          # return default terminal (break recursion)

    if tree_depth<=MAX_DEPTH:                   # if tree max depth it hasn't yet been reached
        idx = gene_seq[i_gene] % 3              # pick a rule

        if idx == 0:                                                                            # 0
            child1 = Node('('+str(i_gene+1)+')cond', parent=node, label='cond')
            child2 = Node('('+str(i_gene+2)+')expr', parent=node, label='expr')
            return "if {cond}:\n{tab}{expr}\n".format(
                                                    cond= cond(gene_seq, child1), tab='\t'*(tree_depth), 
                                                    expr= expr(gene_seq, tree_depth+1, child2)
                                                )
        if idx == 1:                                                                            # 1
            child1 = Node('('+str(i_gene+1)+')cond', parent=node, label='cond')
            child2 = Node('('+str(i_gene+2)+')expr', parent=node, label='expr')
            child3 = Node('('+str(i_gene+3)+')expr', parent=node, label='expr')
            return "if {cond}:\n{tab1}{expr1}\n{tab2}else:\n{tab3}{expr2}\n".format(
                                                                                cond=  cond(gene_seq, child1), tab1='\t'*(tree_depth), 
                                                                                expr1= expr(gene_seq, tree_depth+1, child2), tab2='\t'*(tree_depth-1),  
                                                                                tab3='\t'*(tree_depth), 
                                                                                expr2= expr(gene_seq, tree_depth+1, child3)
                                                                            )
        if idx == 2:                                                                             # 2
            child = Node('('+str(i_gene+1)+')ACTION', parent=node, label='ACT')
            return "action = {act}".format(
                                    act=ACTION(gene_seq, child)
                                )
    else:
        child = Node('('+str(i_gene+1)+')ACTION', parent=node, label='ACT')                     # 2
        return "action = {act}".format(
                                act=ACTION(gene_seq, child)
                            )

def cond(gene_seq, node):
    '''
    <cond>:   "observation[" N_STATES "]" COMP "states[" N_STATES "][" SPLT_PT "]"                               
    '''
    global i_gene
    # i_gene+=1
    if i_gene >= len(gene_seq):
        gene_seq, ext = wrap(gene_seq,'0==0')
        if ext!=None:
            return ext

    child1 = Node('('+str(i_gene+1)+')N_STATES_obser', parent=node, label='N_STATES')                                                                                # 0
    child2 = Node('('+str(i_gene+2)+')COMP', parent=node, label='COMP')
    child3 = Node('('+str(i_gene+1)+')N_STATES_state', parent=node, label='N_STATES')
    child4 = Node('('+str(i_gene+3)+')SPT_PT', parent=node, label='SPLT_PT')

    return "observation[{}] {} states[{}][{}]".format(      
                                                            N_STATES(gene_seq, child1, True),
                                                            COMP(gene_seq, child2), 
                                                            N_STATES(gene_seq, child3, False),
                                                            SPLT_PT(gene_seq, child4)
                                                        )
#-----------TERMINALS-----------------#
def COMP(gene_seq, node):
    '''
    COMP: "<=" | ">"            
    '''
    global i_gene
    i_gene+=1
    if i_gene >= len(gene_seq):
        gene_seq, ext = wrap(gene_seq,'==')
        if ext!=None:
            return ext

    idx = gene_seq[i_gene] % 2
    if idx == 0:
        Node('('+str(i_gene)+')less', parent=node, label='<= ')
        return "<="
    if idx == 1:
        Node('('+str(i_gene)+')great', parent=node, label='> ')
        return ">"


def N_STATES(gene_seq, node, incr):
    '''
    N_STATES: /[0-3]/
    '''
    global i_gene
    if incr:        
        i_gene+=1
        i_gene_same= i_gene
    else:       # must compare a state of the observaion with the same state of all possible observations
        i_gene_same = i_gene -1

    if i_gene_same >= len(gene_seq):
        gene_seq, ext = wrap(gene_seq,'-1')
        if ext!=None:
            return ext
    print(i_gene_same, gene_seq[i_gene_same])
    idx = gene_seq[i_gene_same] % 4
    if idx == 0:
        Node('('+str(i_gene_same)+node.name[-6:]+')idx', parent=node, label='0')
        return "0"
    if idx == 1:
        Node('('+str(i_gene_same)+node.name[-6:]+')idx', parent=node, label='1')
        return "1"
    if idx == 2:
        Node('('+str(i_gene_same)+node.name[-6:]+')idx', parent=node, label='2')
        return "2"
    if idx == 3:
        Node('('+str(i_gene_same)+node.name[-6:]+')idx', parent=node, label='3')
        return "3"


def SPLT_PT(gene_seq, node):
    '''
    SPLIT_PT: /[0-2]/
    '''
    global i_gene
    i_gene+=1
    if i_gene >= len(gene_seq):
        gene_seq, ext = wrap(gene_seq,'-1')
        if ext!=None:
            return ext

    idx = gene_seq[i_gene] % 3
    if idx == 0:
        Node('('+str(i_gene)+')splt', parent=node, label='0')
        return "0"
    if idx == 1:
        Node('('+str(i_gene)+')splt', parent=node, label='1')
        return "1"
    if idx == 2:
        Node('('+str(i_gene)+')splt', parent=node, label='2')
        return "2"


def ACTION(gene_seq, node):
    '''
    ACTION: /[0-1]/
    '''
    global i_gene
    i_gene+=1
    if i_gene >= len(gene_seq):
        gene_seq, ext = wrap(gene_seq,str(np.random.randint(0,1)))
        if ext!=None:
            return ext

    idx = gene_seq[i_gene] % 2
    if idx == 0:
        Node('('+str(i_gene)+')act', parent=node, label='0')
        return "0"
    if idx == 1:
        Node('('+str(i_gene)+')act', parent=node, label='1')
        return "1"
#------------------------------------#
#-------------------------------------------#

#-----------INIT POPULATION-----------------#

int_chromosome = [213, 69, 700, 54, 392, 169, 144, 258, 999, 663]#list(np.random.randint(0,1000,size=CHROMOSOME_LEN))            # sequence of random genes

tree_chromosome = Node('('+str(0)+')expr-start', label='expr')                  # root of derivation tree

program_chromosome="def get_action(observation, states):\n\t"                   # Prepare program whit func def and return value
program_chromosome+= expr(int_chromosome, 2, tree_chromosome)                   # get generated program
program_chromosome+="\n\treturn action"                                         #

file = open('program_chromosome.py', 'w')                                       # Create file and write in generated programs'string
file.write(program_chromosome)                                                  #
file.close()                                                                    #

for pre, _, node in RenderTree(tree_chromosome):                                # print tree on terminal
    print("{}{}".format(pre, node.name))                                        #
RenderTreeGraph(tree_chromosome, nodeattrfunc=lambda node: 'label="{}"'.format( # export tree .png file
    node.label)).to_picture("tree_chromosome.png")                              #


#------------------------------------------#


env = gym.make('CartPole-v0')

#----------PREPARE OBS SPACE---------------#
bins = (3, 3, 3, 3)
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

#------------------------------------------------#

#-----DEFINE CHROMOSOME BASED GET ACTION FUNC-------#
def get_action(observation, states):
    loc={}
    exec(open("./program_chromosome.py").read(), {}, loc)
    action=loc['get_action'](observation, states)
    return action

#------------------------------------------#


observation = env.reset()
try:
    for i,v in enumerate(observation):
        print('observation[{}] = '.format(i),v)
    for i,v in enumerate(states):
        print('states[{}] = '.format(i), v)
    action = get_action(observation, states)
    print('action = ', action)
except:
    print('Failed, Assign low fitness')
    action = env.action_space.sample()
observation, reward, done, info = env.step(action)

