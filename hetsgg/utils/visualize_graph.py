import os
import random

import torch

try:
    from graphviz import Digraph
    HAS_GRAPHVIZ = True
except ImportError:
    HAS_GRAPHVIZ = False
    print("Warning: graphviz not installed. Visualize function will be disabled.")
    # 创建一个空类作为占位符
    class Digraph:
        def __init__(self, *args, **kwargs):
            raise ImportError("graphviz is required for visualization. Install it with: pip install graphviz")


def visual_computation_graph(var, params, output_dir, graph_name='network'):
    """ Produces Graphviz representation of PyTorch autograd graph.

    Blue nodes are trainable Variables (weights, bias).
    Orange node are saved tensors for the backward pass.

    Args:
        var: output Variable
        params: list of (name, Parameters)
    """
    if not HAS_GRAPHVIZ:
        print("Warning: graphviz not available. Skipping visualization.")
        return

    param_map = {id(v): k for k, v in params}

    node_attr = dict(style='filled',
                     shape='box',
                     align='left',
                     fontsize='12',
                     ranksep='0.1',
                     height='0.2')

    comp_graph = Digraph(filename=os.path.join(output_dir, graph_name),
                          format='pdf',
                          node_attr=node_attr,
                          graph_attr=dict(size="256,512"))
    seen = set()



    def get_color():
        pallet = ['#8B0000', "#FF8C00", "#556B2F", "#8FBC8F", "#2F4F4F", "#4682B4",
                  "#191970", "#8A2BE2", "#C71585", "#000000", "#808080"]

        idx = random.randint(0, len(pallet)-1)
        return pallet[idx]


    def add_nodes(var):
        if var not in seen:

            node_id = str(id(var))

            if torch.is_tensor(var):
                node_label = "saved tensor\n{}".format(tuple(var.size()))