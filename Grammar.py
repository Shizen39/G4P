from anytree import Node
import numpy as np

global MAX_RESTART      # max number of time that wrap func is applied to the sequence of genes
global MAX_DEPTH        # max depth of the tree
global restart_ctr      # global counter that count number of time that wrap func is applied
restart_ctr=0
global i_gene            # global index that loops through all genes
i_gene = -1

#-----------DEF GRAMMAR AND TREE/PROGRAM GENERATOR-----------------#
#-----------RULES-----------------#
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
            child1 = Node('('+str(i_gene+1)+')cond', parent=node, label='cond', code="if ")
            cond(gene_seq, child1)
            
            child2 = Node('('+str(i_gene+2)+')expr', parent=node, label='expr', code=":\n{tab1}".format(tab1='\t'*(tree_depth)))
            expr(gene_seq, tree_depth+1, child2)

        if idx == 1:                                                                            # 1
            child1 = Node('('+str(i_gene+1)+')cond', parent=node, label='cond', code="if ")
            cond(gene_seq, child1)
            
            child2 = Node('('+str(i_gene+2)+')expr', parent=node, label='expr', code=":\n{tab1}".format(tab1='\t'*(tree_depth)))
            expr(gene_seq, tree_depth+1, child2)

            child3 = Node('('+str(i_gene+3)+')expr', parent=node, label='expr', code="\n{tab2}else:\n{tab3}".format(tab2='\t'*(tree_depth-1), tab3='\t'*(tree_depth)))
            expr(gene_seq, tree_depth+1, child3)
        if idx == 2:                                                                             # 2
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
        gene_seq, ext = wrap(gene_seq,'0==0')
        if ext!=None:
            return ext
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
        gene_seq, ext = wrap(gene_seq,'==')
        if ext!=None:
            return ext
            
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
        gene_seq, ext = wrap(gene_seq,'-1')
        if ext!=None:
            return ext
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
        gene_seq, ext = wrap(gene_seq,'-1')
        if ext!=None:
            return ext
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
        gene_seq, ext = wrap(gene_seq,str(np.random.randint(0,1)))
        if ext!=None:
            return ext

    idx = gene_seq[i_gene] % 2
    if idx == 0:
        Node('('+str(i_gene)+')act', parent=node, label='0', code="0\n")
    if idx == 1:
        Node('('+str(i_gene)+')act', parent=node, label='1', code="1\n")

#-------FUNCTION UTILITIES--------#
def generate_derivation_tree(gene_seq, root):
    '''
    Starting rule
    '''
    expr(gene_seq, 2, root)
    return root

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