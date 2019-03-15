'''
This file define the Grammar, used to generate chromosomes, as a class that

- define rules of the GRAMMAR as recursive function
    - non-terminal rules are represented by functions that calls other functions
    - terminal rules are represented by functions that don't calls any other functions

- create the DERIVATION TREE selecting derivation rules through a sequence of integer that
    maps the MOD of that integer with a rule.
    A node of the tree contains:
        - id (unique node identifier)
        - parent (parent node, except for the root node)
        - label (rule name)
        - code (encoded python peace of code)

- create the executable PYTHON CODE

'''


from anytree import Node
import numpy as np

class Parser():
    '''
    Grammar parser class. Contains all parameters shared between function calls.

    Args:
        initial_gene_seq (list(int)): the set of genes (integers) of the genotype
        root (AnyTree.Node): starting node of the derivation tre, color=self.color, border=self.bordere
        method (str): method used for generate the tree (full or grow)
        self.MAX_DEPTH (int): maximum depth of the generated phenotypes' derivation trees
        MAX_WRAP  (int): maximum number of time that wrapping operator is applied to genotype
        color (str)
        colorscheme (str)
    '''
    def __init__(self, initial_gene_seq, root, bins, method, MAX_DEPTH, MAX_WRAP):
        self.wrap_ctr = 0                           # global counter that count number of time that wrap func is applied
        self.i_gene = -1                            # global index that loops through all genes
        
        self.initial_gene_seq = initial_gene_seq    # global initial gene_seq (used for wrapping purposes)
        self.root = root                            # starting node

        self.bins = bins

        self.method = method                        # full or grow
        self.MAX_WRAP = MAX_WRAP                    # max number of time that wrap func is applied to the sequence of genes
        self.MAX_DEPTH = MAX_DEPTH -2               # max depth of the tree

        self.color=''
        self.border=''
        self.extra_id=''
        
        
    #-------FUNCTION UTILITIES--------#
    def start_derivating(self, node_type, tree_depth=0, indent=1, extra_id=''): 
        ''' Starting derivation rule. '''
        self.extra_id = extra_id
        self.color = self.root.color
        self.border = self.root.border
        if node_type=='expr':
            self.expr(self.initial_gene_seq, tree_depth, self.root, indent)
        if node_type=='cond':
            self.cond(self.initial_gene_seq, self.root)
        return self.root

    def wrap(self, gene_seq, is_terminal):
        '''
        Function called when translation from genes to grammars' rules runs out of genes
        '''
        if is_terminal:
            # if it's translating a terminal, finish the translation wrapping gene_seq by only 1 element
            # so the next time (if it's not another terminal and if MAX_WRAP is reached, it will stop wrapping)
            gene_seq += self.initial_gene_seq[len(gene_seq) % len(self.initial_gene_seq)],
        else:
            # if it's translating a non-terminal, increment the counter
            self.wrap_ctr+=1
            if self.wrap_ctr>self.MAX_WRAP:
                # if wrapped too many times, extend gene_seq by a gene that will always lead to a terminal
                gene_seq+=[3]
            else:
                #if can wrap, extend actual gene_seq by the initial one
                gene_seq += self.initial_gene_seq
        return gene_seq


    #-----------DEF GRAMMAR AND TREE/PROGRAM GENERATOR-----------------#
    #-----------RULES-----------------#
    def expr(self, gene_seq, tree_depth, node, indent):
        '''
        <expr>:   "if" <cond> ":" _NL _INDENT <expr> _NL                                            # 0
                | "if" <cond> ":" _NL _INDENT <expr> _NL "else:"_NL _INDENT <expr> _NL              # 1
                | <expr> _NL _INDENT <expr>                                                         # 2
                | "action =" ACTION                                                                 # 3
        '''
        self.i_gene+=1
        if self.i_gene >= len(gene_seq):                 # if translation from genes to grammrs' rules runned out of genes (self.i_gene is the index of genes)
            gene_seq = self.wrap(gene_seq, False)
            
        if tree_depth<self.MAX_DEPTH:                    # if tree max depth it hasn't yet been reached
            if self.method == 'full':                    # and method is FULL
                idx = gene_seq[self.i_gene] % 3          # skip terminal ACTION so we have always tree of max_depth
            else:                                        # method is GROW
                if node.name.rsplit(')')[1].rsplit('_id')[0]=='expr_a' or node.name.rsplit(')')[1].rsplit('_id')[0]=='expr_b':
                    idx = gene_seq[self.i_gene] % 3 # in order to not have two ACTION terminals that aren't a consequence of if/else in case <expr><expr> was chosen
                else:
                    idx = gene_seq[self.i_gene] % 4

            i_gene = self.i_gene
            if idx == 0:                                                                            # 0
                child1 = Node('('+str(i_gene)+')cond'+'_id_'+str(id(node)), parent=node, label='cond', code="if ", color=self.color, border=self.border)
                self.cond(gene_seq, child1)
                
                child2 = Node('('+str(i_gene)+')expr_i'+'_id_'+str(id(node)), parent=node, label='expr', code=":\n{tab1}".format(tab1='\t'*(indent+1)), indent=indent+1, color=self.color, border=self.border)
                self.expr(gene_seq, tree_depth+1, child2, indent+1)

            if idx == 1:                                                                            # 1
                child1 = Node('('+str(i_gene)+')cond'+'_id_'+str(id(node)), parent=node, label='cond', code="if ", color=self.color, border=self.border)
                self.cond(gene_seq, child1)
                
                child2 = Node('('+str(i_gene)+')expr_i'+'_id_'+str(id(node)), parent=node, label='expr', code=":\n{tab1}".format(tab1='\t'*(indent+1)), indent=indent+1, color=self.color, border=self.border)
                self.expr(gene_seq, tree_depth+1, child2, indent+1)

                child3 = Node('('+str(i_gene)+')expr_e'+'_id_'+str(id(node)), parent=node, label='expr', code="\n{tab2}else:\n{tab3}".format(tab2='\t'*(indent), tab3='\t'*(indent+1)), indent=indent+1, color=self.color, border=self.border)
                self.expr(gene_seq, tree_depth+1, child3, indent+1)
            if idx == 2:                                                                            # 2
                child1 = Node('('+str(i_gene)+')expr_a'+'_id_'+str(id(node)), parent=node, label='expr', code="", indent=indent, color=self.color, border=self.border)
                self.expr(gene_seq, tree_depth+1, child1, indent)

                child2 = Node('('+str(i_gene)+')expr_b'+'_id_'+str(id(node)), parent=node, label='expr', code="\n{tab1}".format(tab1='\t'*(indent)), indent=indent, color=self.color, border=self.border)
                self.expr(gene_seq, tree_depth+1, child2, indent)
            if idx == 3:                                                                             # 3
                child = Node('('+str(i_gene)+')ACTION'+'_id_'+str(id(node)), parent=node, label='ACT', code="action = ", color=self.color, border=self.border)
                self.ACTION(gene_seq, child)
        else:
            child = Node('('+str(self.i_gene)+')ACTION'+'_id_'+str(id(node)), parent=node, label='ACT', code="action = ", color=self.color, border=self.border)
            self.ACTION(gene_seq, child)


    def cond(self, gene_seq, node):
        '''
        <cond>:   "observation[" N_STATES "]" COMP "states[" N_STATES "][" SPLT_PT "]"                               
        '''
        if self.i_gene >= len(gene_seq):
            gene_seq = self.wrap(gene_seq, True)
        i_gene=self.i_gene
        child1 = Node('('+str(i_gene)+')N_STATES_obser'+'_id_'+str(id(node)), parent=node, label='N_STATES', code="observation[", color=self.color, border=self.border)
        self.N_STATES(gene_seq, child1, True, '_obser')
            
        child2 = Node('('+str(i_gene)+')COMP'+'_id_'+str(id(node)), parent=node, label='COMP', code='] ', color=self.color, border=self.border)
        self.COMP(gene_seq, child2)

        child3 = Node('('+str(i_gene)+')N_STATES_state'+'_id_'+str(id(node)), parent=node, label='N_STATES', code=' states[', color=self.color, border=self.border)
        n_states = self.N_STATES(gene_seq, child3, False, '_state')
        
        child4 = Node('('+str(i_gene)+')SPT_PT'+'_id_'+str(id(node)), parent=node, label='SPLT_PT', code='][', color=self.color, border=self.border)
        self.SPLT_PT(gene_seq, child4, n_states)       


    #-----------TERMINALS-----------------#
    def COMP(self, gene_seq, node):
        '''
        COMP: "<=" | ">"            
        '''
        self.i_gene+=1
        if self.i_gene >= len(gene_seq):
            gene_seq = self.wrap(gene_seq, True)
        
        idx = gene_seq[self.i_gene] % 2
        if idx == 0:
            Node('('+str(self.i_gene)+')less'+'_id_'+str(id(node)), parent=node, label='<= ', code="<=", color=self.color, border=self.border)
        if idx == 1:
            Node('('+str(self.i_gene)+')great'+'_id_'+str(id(node)), parent=node, label='> ', code=">", color=self.color, border=self.border)


    def N_STATES(self, gene_seq, node, incr, arr):
        '''
        N_STATES: /[0-3]/
        '''
        # must compare a state of the observaion with the same state of all possible observations
        # so the two N_STATES must have the same i_seq
        if incr:        
            self.i_gene+=1
            i_gene_same= self.i_gene
        else:
            i_gene_same = self.i_gene -1

        if i_gene_same >= len(gene_seq):
            gene_seq = self.wrap(gene_seq, True)
        
        idx = gene_seq[i_gene_same] % 4
        if idx == 0:
            Node('('+str(i_gene_same)+arr+')idx'+'_id_'+str(id(node)), parent=node, label='0', code="0", color=self.color, border=self.border)
        if idx == 1:
            Node('('+str(i_gene_same)+arr+')idx'+'_id_'+str(id(node)), parent=node, label='1', code="1", color=self.color, border=self.border)
        if idx == 2:
            Node('('+str(i_gene_same)+arr+')idx'+'_id_'+str(id(node)), parent=node, label='2', code="2", color=self.color, border=self.border)
        if idx == 3:
            Node('('+str(i_gene_same)+arr+')idx'+'_id_'+str(id(node)), parent=node, label='3', code="3", color=self.color, border=self.border)
        return idx


    def SPLT_PT(self, gene_seq, node, n_states):
        '''
        SPLIT_PT: /[0-2]/
        '''
        self.i_gene+=1
        if self.i_gene >= len(gene_seq):
            gene_seq = self.wrap(gene_seq, True)
        
        idx = gene_seq[self.i_gene] % self.bins[n_states]

        for n_of_splt in range(self.bins[n_states]):
            if idx == n_of_splt:
                Node('('+str(self.i_gene)+')splt'+'_id_'+str(id(node)), parent=node, label=str(idx), code=str(idx)+"]", color=self.color, border=self.border)


    def ACTION(self, gene_seq, node):
        '''
        ACTION: /[0-1]/
        '''
        self.i_gene+=1
        if self.i_gene >= len(gene_seq):
            gene_seq = self.wrap(gene_seq, True)
        
        idx = gene_seq[self.i_gene] % 2
        if idx == 0:
            Node('('+str(self.i_gene)+')act'+'_id_'+str(id(node)), parent=node, label='0', code="0\n", color=self.color, border=self.border)
        if idx == 1:
            Node('('+str(self.i_gene)+')act'+'_id_'+str(id(node)), parent=node, label='1', code="1\n", color=self.color, border=self.border)