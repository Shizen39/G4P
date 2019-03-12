from anytree import Node, RenderTree
from anytree.exporter import DotExporter
from anytree.dotexport import RenderTreeGraph
from anytree import PreOrderIter
import numpy as np
import os


def tree_to_png(generation, root):
    if not os.path.exists('./outputs/{}'.format('self')):
        os.mkdir('./outputs/{}'.format('self'))
    DotExporter(root, 
        nodeattrfunc=lambda node: 'label="{}", style=filled, color={}, fillcolor={}'.format(node.label, node.border, node.color),
        edgeattrfunc=lambda node,child: 'color={}'.format(node.border)
        ).to_picture("./outputs/{}/GEN-{}.png".format('self', generation))
    


root = Node('('+str(0)+')expr-start', label='expr', code='', color='/greys9/1', border='/greys9/9')

tree_to_png(0, root)