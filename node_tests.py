import numpy as np
from anytree import Node, PreOrderIter, RenderTree
from anytree.dotexport import RenderTreeGraph
from anytree.exporter import DotExporter
from anytree.search import find

root = Node('('+str(0)+')expr-start', label='expr', code='')
child1 = Node('('+str(1)+')expr_a', parent=root, label='expr', code="")
child2 = Node('('+str(2)+')expr_b', parent=root, label='expr', code="\n{tab1}".format(tab1='\t'*(1-1)))

child1child1 = Node('('+str(3)+')cond', parent=child1, label='cond', code="if ")
child1child2 = Node('('+str(4)+')expr', parent=child1, label='expr', code=":\n{tab1}".format(tab1='\t'*(1)))
child1child2child = Node('('+str(5)+')ACTION', parent=child1child2, label='ACT', code="action = ")

child2child1 = Node('('+str(6)+')cond', parent=child2, label='cond', code="if ")
child2child2 = Node('('+str(7)+')expr', parent=child2, label='expr', code=":\n{tab1}".format(tab1='\t'*(1)))
child2child2child = Node('('+str(8)+')ACTION', parent=child2child2, label='ACT', code="action = ")

RenderTreeGraph(root, nodeattrfunc=lambda node: 'label="{}"'.format( # export tree .png file
                node.label)).to_picture("test_not_mutated.png")



#retrieve max node id
childs_node = root.children
while childs_node[-1] !=None:
    childs_node = childs_node[-1].children
    if childs_node[-1].is_leaf:
        childs_node = childs_node[-1]
        break


    
max_id_for_tree = int(''.join(filter(str.isdigit, childs_node.name)))
#generate random id that will be the random picked node
print(max_id_for_tree)
rand_id = 1#np.random.randint(max_id_for_tree)
print(rand_id)
#list of nodes names that can be picked (only non-terminals) 
#MAYBE ALSO COND???
non_terminals = ['({})expr'.format(rand_id), '({})expr_a'.format(rand_id), '({})expr_b'.format(rand_id),
    '({})mutexpr'.format(rand_id), '({})mutexpr_a'.format(rand_id), '({})mutexpr_b'.format(rand_id),]

#find that node
x = find(root, lambda node: node.name in non_terminals)
print(x)
id_of_x = int(''.join(filter(str.isdigit, x.name)))
#create random subtree
mutation = Node('('+str(id_of_x)+')mutexpr_b', label='expr-mutated', code="")
mchild1 = Node('('+str(id_of_x+1)+')mutcond', parent=mutation, label='mucond', code="if ")
mchild2 = Node('('+str(id_of_x+2)+')mutexpr', parent=mutation, label='mutexpr', code=":\n{tab1}".format(tab1='\t'*(1)))
mchild2child = Node('('+str(id_of_x+3)+')mutACTION', parent=mchild2, label='mutACT', code="action = ")
#attach mutated root to parent of x (appending it at the x parents' children list)
#detach x (by modifing its parents' children list and removing it from there)
tmp_children = list(x.parent.children)
tmp_children[x.parent.children.index(x)] = mutation
x.parent.children = tuple(tmp_children)

#detach x (by modifing its parents' children list and removing it from there)


RenderTreeGraph(root, nodeattrfunc=lambda node: 'label="{}"'.format( # export tree .png file
                node.label)).to_picture("test_mutated.png")



