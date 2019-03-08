from anytree import Node
import numpy as np
    
global MAX_WRAP         # max number of time that wrap func is applied to the sequence of genes
global MAX_DEPTH        # max depth of the tree

global wrap_ctr         # global counter that count number of time that wrap func is applied
wrap_ctr=0
global i_gene           # global index that loops through all genes
i_gene = -1

global initial_gene_seq # global initial gene_seq (used for wrapping purposes)
global method

#-----------DEF GRAMMAR AND TREE/PROGRAM GENERATOR-----------------#
#-----------RULES-----------------#
def expr(gene_seq, tree_depth, node, indent):
    '''
    <expr>:   "if" <cond> ":" _NL _INDENT <expr> _NL                                            # 0
            | "if" <cond> ":" _NL _INDENT <expr> _NL "else:"_NL _INDENT <expr> _NL              # 1
            | <expr> _NL _INDENT <expr>                                                         # 2
            | "action =" ACTION                                                                 # 3
    '''
    global i_gene
    global method
    i_gene+=1
    if i_gene >= len(gene_seq):                 # if translation from genes to grammrs' rules runned out of genes (i_gene is the index of genes)
        gene_seq = wrap(gene_seq, False)
        
    if tree_depth<MAX_DEPTH:                    # if tree max depth it hasn't yet been reached
        if method == 'full':
            idx = gene_seq[i_gene] % 3          # skip terminal ACTION
        else:
            idx = gene_seq[i_gene] % 3 if node.name=='('+str(i_gene+1)+')expr1' or node.name=='('+str(i_gene+1)+')expr2' \
                else gene_seq[i_gene] % 4       # in order to not have two ACTION terminals that aren't a consequence of if/else in case <expr><expr> was chosen

        if idx == 0:                                                                            # 0
            child1 = Node('('+str(i_gene+1)+')cond', parent=node, label='cond', code="if ")
            cond(gene_seq, child1)
            
            child2 = Node('('+str(i_gene+2)+')expr', parent=node, label='expr', code=":\n{tab1}".format(tab1='\t'*(indent)))
            expr(gene_seq, tree_depth+1, child2, indent+1)

        if idx == 1:                                                                            # 1
            child1 = Node('('+str(i_gene+1)+')cond', parent=node, label='cond', code="if ")
            cond(gene_seq, child1)
            
            child2 = Node('('+str(i_gene+2)+')expr', parent=node, label='expr', code=":\n{tab1}".format(tab1='\t'*(indent)))
            expr(gene_seq, tree_depth+1, child2, indent+1)

            child3 = Node('('+str(i_gene+3)+')expr', parent=node, label='expr', code="\n{tab2}else:\n{tab3}".format(tab2='\t'*(indent-1), tab3='\t'*(indent)))
            expr(gene_seq, tree_depth+1, child3, indent+1)
        if idx == 2:
            child1 = Node('('+str(i_gene+1)+')expr1', parent=node, label='expr', code="")
            expr(gene_seq, tree_depth+1, child1, indent)

            child2 = Node('('+str(i_gene+2)+')expr2', parent=node, label='expr', code="\n{tab1}".format(tab1='\t'*(indent-1)))
            expr(gene_seq, tree_depth+1, child2, indent)
        if idx == 3:                                                                             # 2
            child = Node('('+str(i_gene+1)+')ACTION', parent=node, label='ACT', code="action = ")
            ACTION(gene_seq, child)
    else:
        child = Node('('+str(i_gene+1)+')ACTION', parent=node, label='ACT', code="action = ")
        ACTION(gene_seq, child)


def cond(gene_seq, node):
    '''
    <cond>:   "observation[" N_STATES "]" COMP "states[" N_STATES "][" SPLT_PT "]"                               
    '''
    global i_gene
    if i_gene >= len(gene_seq):
        gene_seq = wrap(gene_seq, True)
    
    child1 = Node('('+str(i_gene+1)+')N_STATES_obser', parent=node, label='N_STATES', code="observation[")  
    N_STATES(gene_seq, child1, True)
        
    child2 = Node('('+str(i_gene+2)+')COMP', parent=node, label='COMP', code='] ')
    COMP(gene_seq, child2)

    child3 = Node('('+str(i_gene+1)+')N_STATES_state', parent=node, label='N_STATES', code=' states[')
    N_STATES(gene_seq, child3, False)
    
    child4 = Node('('+str(i_gene+3)+')SPT_PT', parent=node, label='SPLT_PT', code='][')
    SPLT_PT(gene_seq, child4)       


#-----------TERMINALS-----------------#
def COMP(gene_seq, node):
    '''
    COMP: "<=" | ">"            
    '''
    global i_gene
    i_gene+=1
    if i_gene >= len(gene_seq):
        gene_seq = wrap(gene_seq, True)
    
    idx = gene_seq[i_gene] % 2
    if idx == 0:
        Node('('+str(i_gene)+')less', parent=node, label='<= ', code="<=")
    if idx == 1:
        Node('('+str(i_gene)+')great', parent=node, label='> ', code=">")


def N_STATES(gene_seq, node, incr):
    '''
    N_STATES: /[0-3]/
    '''
    global i_gene
    # must compare a state of the observaion with the same state of all possible observations
    # so the two N_STATES must have the same i_seq
    if incr:        
        i_gene+=1
        i_gene_same= i_gene
    else:
        i_gene_same = i_gene -1

    if i_gene_same >= len(gene_seq):
        gene_seq = wrap(gene_seq, True)
     
    idx = gene_seq[i_gene_same] % 4
    if idx == 0:
        Node('('+str(i_gene_same)+node.name[-6:]+')idx', parent=node, label='0', code="0")
    if idx == 1:
        Node('('+str(i_gene_same)+node.name[-6:]+')idx', parent=node, label='1', code="1")
    if idx == 2:
        Node('('+str(i_gene_same)+node.name[-6:]+')idx', parent=node, label='2', code="2")
    if idx == 3:
        Node('('+str(i_gene_same)+node.name[-6:]+')idx', parent=node, label='3', code="3")


def SPLT_PT(gene_seq, node):
    '''
    SPLIT_PT: /[0-2]/
    '''
    global i_gene
    i_gene+=1
    if i_gene >= len(gene_seq):
        gene_seq = wrap(gene_seq, True)
     
    idx = gene_seq[i_gene] % 3
    if idx == 0:
        Node('('+str(i_gene)+')splt', parent=node, label='0', code="0]")
    if idx == 1:
        Node('('+str(i_gene)+')splt', parent=node, label='1', code="1]")
    if idx == 2:
        Node('('+str(i_gene)+')splt', parent=node, label='2', code="2]")


def ACTION(gene_seq, node):
    '''
    ACTION: /[0-1]/
    '''
    global i_gene
    i_gene+=1
    if i_gene >= len(gene_seq):
        gene_seq = wrap(gene_seq, True)
    
    idx = gene_seq[i_gene] % 2
    if idx == 0:
        Node('('+str(i_gene)+')act', parent=node, label='0', code="0\n")
    if idx == 1:
        Node('('+str(i_gene)+')act', parent=node, label='1', code="1\n")

#-------FUNCTION UTILITIES--------#
def start_derivating(gene_seq, root, _method, _MAX_DEPTH, _MAX_WRAP, _initial_gene_seq):
    '''
    Starting rule
    '''
    global method, MAX_DEPTH, MAX_WRAP, initial_gene_seq
    method = _method
    MAX_DEPTH = _MAX_DEPTH - 2   # max depth of the tree
    MAX_WRAP = _MAX_WRAP         # max number of time wrapping operator is applied
    initial_gene_seq = _initial_gene_seq

    expr(gene_seq, 0, root, 2)
    return root

def wrap(gene_seq, is_terminal):
    '''
    Function called when translation from genes to grammrs' rules runs out of genes
    '''
    if is_terminal:
        # if it's translating a terminal, finish the translation wrapping gene_seq by only 1 element
        # so the next time (if it's not another terminal and if MAX_WRAP is reached, it will stop wrapping)
        gene_seq += initial_gene_seq[22%len(initial_gene_seq)],
    else:
        # if it's translating a non-terminal, increment the counter
        global wrap_ctr
        wrap_ctr+=1
        if wrap_ctr>MAX_WRAP:
            # if wrapped too many times, extend gene_seq by a gene that will always lead to a terminal
            gene_seq+=[3]
        else:
            #if can wrap, extend actual gene_seq by the initial one
            print('Wrapping!')
            gene_seq += initial_gene_seq
    return gene_seq