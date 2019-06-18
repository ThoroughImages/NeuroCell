from collections import defaultdict, Counter, OrderedDict
from copy import copy, deepcopy
from itertools import chain
import logging
import random
import time
import sys

import numpy as np
import pdb
import torch
import torch.nn as nn

from config import ACTIVATION, BATCHNORM


class Node(object):

    def __init__(self, name, layer, op_type=None):
        self.name = name
        self.layer = layer
        self.op_type = op_type
        self.output_trace = None

        self.prev_list = []
        self.next_list = []

    def __repr__(self):
        return_str = """
                        Name: {0}
                        Previous: {1}
                        Next: {2}
                        =========================                                                        
                     """.format(self.name, [node.name for node in self.prev_list], [node.name for node in self.next_list])

        return return_str


class Graph(nn.Module):

    def __init__(self, layer_dict):
        super().__init__()
        self.layer_dict = layer_dict
        self.node_dict = {name: Node(name, layer, Graph.__get_op_type(name)) for name, layer in layer_dict.items()}
        self.wider_out_channels_visited = {nodename: False for nodename in self.node_dict}
        self.wider_in_channels_visited = {nodename: False for nodename in self.node_dict}

    def __add__(self, h):
        combined_graph = Graph(nn.ModuleDict({}))

        # Merge layer_dict's
        for layername, layer in self.layer_dict.items():
            combined_graph.layer_dict[layername] = layer
        for layername, layer in h.layer_dict.items():
            combined_graph.layer_dict[layername] = layer

        # Merge node_dict's
        for nodename, node in self.node_dict.items():
            combined_graph.node_dict[nodename] = node
        for nodename, node in h.node_dict.items():
            combined_graph.node_dict[nodename] = node

        # Merge wider visits's
        for nodename, node in self.node_dict.items():
            combined_graph.wider_out_channels_visited[nodename] = False
            combined_graph.wider_in_channels_visited[nodename] = False
        for nodename, node in h.node_dict.items():
            combined_graph.wider_out_channels_visited[nodename] = False
            combined_graph.wider_in_channels_visited[nodename] = False

        # Connecting two graphs
        self.node_dict[self.tailname].next_list, h.node_dict[h.headname].prev_list = [h.node_dict[h.headname]], [self.node_dict[self.tailname]]

        return combined_graph

    @staticmethod
    def topological_sort(node_dict):
        def helper(node, visited_dict, stack):
            visited_dict[node.name] = True

            for n in node.next_list:
                try:
                    visited = visited_dict[n.name]
                except KeyError:
                    continue
                else:
                    if not visited:
                        helper(n, visited_dict, stack)

            stack.append(node.name)

        visited_dict = {name: False for name, node in node_dict.items()}
        stack = []

        for nodename, node in node_dict.items():
            if not visited_dict[nodename]:
                helper(node, visited_dict, stack)

        return stack

    def get_num_layers(self):
        assert len(self.layer_dict) == len(self.node_dict), "Number of layers does not match number of nodes!"

        return len(self.layer_dict)

    @property
    def headname(self):
        node_stack = self.topological_sort(self.node_dict)

        return node_stack[-1]

    @property
    def tailname(self):
        node_stack = self.topological_sort(self.node_dict)

        return node_stack[0]

    def add_edge(self, node_name1, node_name2):
        self.node_dict[node_name1].next_list.append(self.node_dict[node_name2])
        self.node_dict[node_name2].prev_list.append(self.node_dict[node_name1])

    def add_edge_from_layer_list(self, layer_list):
        for i in range(len(layer_list) - 1):
            self.add_edge(layer_list[i][0], layer_list[i + 1][0])

    def clear_all_in_out_visited_channels(self):
        for nodename in self.node_dict:
            self.wider_out_channels_visited[nodename] = False
            self.wider_in_channels_visited[nodename] = False

    def add_new_node_to_graph(self, newname, newnode):
        self.node_dict[newname] = newnode
        self.layer_dict[newname] = newnode.layer
        self.wider_out_channels_visited[new_name] = False
        self.wider_in_channels_visited[new_name] = False

    def connect_nodes(self, node1, node2):
        """
        This method connects two nodes together where node1 points to node2.
        """
        node1.next_list.append(node2)
        node2.prev_list.append(node1)

    def forward(self, x):
        process_stack = self.topological_sort(self.node_dict)

        while process_stack:
            nodename = process_stack.pop()
            current_node = self.node_dict[nodename]

            if not current_node.prev_list:
                x = current_node.layer(x)
            else:
                if Graph.__get_op_type(nodename) != 'cat':
                    try:
                        out_trace_sum = sum([node.output_trace for node in current_node.prev_list])
                    except RuntimeError:
                        print("Node name: {}".format(current_node.name))
                        print("Summing up inputs from: {}".format(current_node.prev_list))
                        print("Input shapes: {}".format([node.output_trace.shape for node in current_node.prev_list]))
                        sys.exit(1)

                    x = current_node.layer(out_trace_sum)
                else:
                    x = concat([node.output_trace for node in current_node.prev_list])

            current_node.output_trace = x

        return x

    def clear_feature_maps(self):
        for nodename, node in self.node_dict.items():
            node.output_trace = None

    def _is_simple(self, nodename):
        assert len(self.node_dict.keys()) == len(self.layer_dict), "node_dict and layer_dict not in sync!"

        node = self.node_dict[nodename]

        if len(node.prev_list) < 2 and len(node.next_list) < 2:
            return True

        else:
            return False 

    def get_random_layername(self, layer_type=None, random_seed=None):
        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)
            torch.manual_seed(random_seed)

        while True:
            if layer_type is None:
               layername = random.choice(list(self.layer_dict.keys()))
            else:
                layername = random.choice([layername for layername in self.layer_dict.keys() if layer_type in layername])

            if 'conv' in layername or 'fc' in layername:
                return layername

    def mutate(self, mutation_type, cell=None, layername=None, nc=True, random_seed=None, **kwargs):
        """
        Three conditions have to be met after each mutation:
        1. The graph internal structure has to be maintained (how nodes are connected).
        2. The node dict of the graph has to be renewed so that any new generated nodes are tracked by the node dict.
        3. The layer dict of the graph has to be renewed so that the optimizer can update the parameters of those nodes.
        4. Each node need to have a unique name that are tracked by both the node dict and the layer dict.
        """
        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)
            torch.manual_seed(random_seed)

        def wider(nodename, nc, **kwargs):
            # Get old layer
            layer = self.layer_dict[nodename]

            # Get old width
            if kwargs:
                new_width = kwargs['new_width']
            else:
                old_width = layer.weight.data.size(0)
                widen_factor = np.random.uniform(low=1.2, high=2)
                new_width = round(old_width * widen_factor)

                if new_width == old_width:
                    new_width += 1

            _ = self.make_wider(nodename, new_width, nc)

            # Reset status
            self.wider_out_channels_visited = {nodename: False for nodename, node in self.node_dict.items()}
            self.wider_in_channels_visited = {nodename: False for nodename, node in self.node_dict.items()}

            return new_width

        def deeper(nodename, nc, **kwargs):
            activation = nn.ReLU if ACTIVATION else None
            batchnorm_flag = BATCHNORM

            new_layername = self.make_deeper(nodename, nc, activation=activation, batchnorm_flag=batchnorm_flag, **kwargs)

            return new_layername

        def wider_kernel(nodename, nc):
            if 'conv' in nodename:
                _ = self.widen_kernel(nodename, nc)
            else:
                print("Mutation type doesn't apply to layer {}, repicking mutation type...".format(nodename))
                mutation_type = np.random.choice(list(mutation_funcs.keys()))
                print("Mutation type is now: {}".format(mutation_type))
                _ = mutation_logic(nodename, mutation_type, nc)

        def add_skip_connection(nodename, nc, **kwargs):
            if 'conv' in nodename:
                activation = nn.ReLU if ACTIVATION else None
                batchnorm_flag = BATCHNORM

                new_layername = self.add_skip_connection(nodename, nc, activation=activation, batchnorm_flag=batchnorm_flag,**kwargs)

                return new_layername
            else:
                print("Mutation type doesn't apply to layer {}, repicking mutation type...".format(nodename))
                mutation_type = np.random.choice(list(mutation_funcs.keys()))
                print("Mutation type is now: {}".format(mutation_type))
                _ = mutation_logic(nodename, mutation_type, nc)

        def mutation_logic(layername, mutation_type, nc, **kwargs):
            if mutation_type in {'deeper', 'skip_connection'}:
                new_layer_list = mutation_funcs[mutation_type](layername, nc, **kwargs)

                return new_layer_list

            elif mutation_type == 'wider':
                new_width = mutation_funcs[mutation_type](layername, nc, **kwargs)

                return new_width

            else:
                _ = mutation_funcs[mutation_type](layername, nc, **kwargs)

        mutation_funcs = {'wider': wider, 'deeper': deeper, 'wider_kernel': wider_kernel,
                          'skip_connection': add_skip_connection}

        if cell:
            if not layername:
                layername = cell.get_random_layername()
            else:
                if layername not in cell.node_dict:
                    print("Layername: {}".format(layername))
                    print("Layername Dict: {}".format(cell.node_dict))
                    raise ValueError("The specified layer is not in the cell.")

            if mutation_type in {'deeper', 'skip_connection'}:
                new_layername = mutation_logic(layername, mutation_type, nc, **kwargs)  #new_name, activation_name, bnorm_name

                if new_layername:
                   cell.node_dict[new_layername] = self.node_dict[new_layername]
                   cell.layer_dict[new_layername] = self.layer_dict[new_layername]

                   split_idx = layername.index('_') 

                   return layername[split_idx + 1:], new_layername

            elif mutation_type == 'wider':
                new_width = mutation_logic(layername, mutation_type, nc, **kwargs)

                split_idx = layername.index('_') 

                return layername[split_idx + 1:], new_width

            else:
                _ = mutation_logic(layername, mutation_type, nc, **kwargs)

                split_idx = layername.index('_') 

                return layername[split_idx + 1:]


            new_layername = mutation_logic(layername, mutation_type, nc, **kwargs)  #new_name, activation_name, bnorm_name

        else:
            if not layername:
                layername = self.get_random_layername()
            else:
                if layername not in self.node_dict:
                    raise ValueError("The specified layer is not in the model.")

            _ = mutation_logic(layername, mutation_type, nc)

        self.clear_all_in_out_visited_channels()

    def _find_downstream_layers(self, layername):
        current_node = self.node_dict[layername]
        layername_list = []

        for node in current_node.next_list:
            layername_list.append(node.name)

            if 'conv' not in node.name and 'fc' not in node.name:
                layername_list += self._find_downstream_layers(node.name)

        return list(set(layername_list))

    def _find_upstream_layers(self, layername):
        current_node = self.node_dict[layername]
        layername_list = []

        for node in current_node.prev_list:
            layername_list.append(node.name)

            if 'conv' not in node.name and 'fc' not in node.name:
                layername_list += self._find_upstream_layers(node.name)

        return list(set(layername_list))

    def make_wider(self, layername1, new_width, nc):
        if self.wider_out_channels_visited[layername1]:
            return

        downstream_layer_list = self._find_downstream_layers(layername1)
        downstream_conv_fc_list = [layername for layername in downstream_layer_list if 'conv' in layername or 'fc' in layername]
        downstream_bnorm_list = [layername for layername in downstream_layer_list if 'batchnorm' in layername]

        if not downstream_conv_fc_list + downstream_bnorm_list:
            return None

        layer1 = self.node_dict[layername1].layer

        # Get layer1's weight and bias
        weight1, bias1 = layer1.weight.data, layer1.bias.data

        # Get the original shape of layer1, aka the input size and the original width
        layer1_num_input_units, layer1_num_output_units = weight1.size(1), weight1.size(0)
        old_width = layer1_num_output_units

        ## Check if the new width is wider
        if old_width > new_width:
            raise ValueError("New width has to be larger than current width.")

        # Placeholders for the widened weight/bias (layer1)
        weight1_widened, bias1_widened = weight1.clone(), bias1.clone()

        # Prepare the g function
        g_mapping = {j: j if j < old_width else np.random.choice(old_width) for j in range(new_width)}
        g_counter = Counter(g_mapping.values())

        if "Conv" in layer1.__class__.__name__:  # When current layer is conv
            # Check current weight rank
            assert weight1.dim() == 4
            w1_kh_size, w1_kw_size = weight1.size(2), weight1.size(3)

            # Widen weight1/bias1
            weight1_widened.resize_(new_width, layer1_num_input_units, w1_kh_size, w1_kw_size)
            bias1_widened.resize_(new_width)

            # Intialize the extended parts with zeros
            weight1_widened[old_width:new_width, :, :, :] = 0

            for j in range(old_width, new_width):
                weight1_widened[j, :, :, :] = weight1[g_mapping[j], :, :, :]
                bias1_widened[j] = bias1[g_mapping[j]]

            # Add small noise to break symmetry
            weight1_widened[old_width:new_width, :, :, :] = weight1_widened[old_width:new_width, :, :, :] \
                                                            + torch.randn(weight1_widened[old_width:new_width, :, :, :].shape) * 1e-5

            # Update weights for layer 1
            layer1.weight, layer1.bias = nn.Parameter(weight1_widened), nn.Parameter(bias1_widened)

            # Update layer 1 properties
            layer1.out_channels = layer1.weight.data.size(0)

            for layername in downstream_conv_fc_list:
                if self.wider_in_channels_visited[layername]:
                    continue

                layer2 = self.node_dict[layername].layer

                # Get layer2's weight and bias
                weight2 = layer2.weight.data

                # Get the original shape of layer2, aka the input size and the original width
                layer2_num_input_units, layer2_num_output_units = weight2.size(1), weight2.size(0)

                # Placeholders for the widened weight/bias (layer2)
                weight2_widened = weight2.clone()

                if "Conv" in layer2.__class__.__name__:  # When next layer is conv
                    ## Check if the output channels of layer 1 matches the input channels of layer 2
                    # Check next weight rank
                    assert weight2.dim() == 4
                    w2_kh_size, w2_kw_size = weight2.size(2), weight2.size(3)

                    # Widen weight2
                    weight2_widened.resize_(layer2_num_output_units, new_width, w2_kh_size, w2_kw_size)
                    # Intialize the extended parts with zeros
                    weight2_widened[:, old_width:new_width, :, :] = 0

                    for j in range(new_width):
                        weight2_widened[:, j, :, :] = weight2[:, g_mapping[j], :, :] / g_counter[g_mapping[j]]

                    # Update weights for layer 2
                    layer2.weight = nn.Parameter(weight2_widened)

                    # Update layer 2 properties
                    layer2.in_channels = layer2.weight.data.size(1)

                    # Update tail
                    self.node_dict[layername].layer = layer2

                elif "Linear" in layer2.__class__.__name__:  # When next layer is linear
                    # Check next weight rank
                    assert weight2.dim() == 2

                    # Get feature map size and side length
                    feature_map_size = layer2_num_input_units // layer1_num_output_units
                    feature_map_side_length = int(np.sqrt(feature_map_size))

                    weight2_reshaped = weight2_widened.reshape(layer2_num_output_units, layer1_num_output_units, \
                                                               feature_map_side_length, feature_map_side_length)
                    weight2_widened = weight2_widened.reshape(layer2_num_output_units, layer1_num_output_units, \
                                                              feature_map_side_length, feature_map_side_length).clone()

                    # Widen weight2
                    weight2_widened.resize_(layer2_num_output_units, new_width, feature_map_side_length,
                                            feature_map_side_length)

                    # Intialize the extended parts with zeros
                    weight2_widened[:, old_width:new_width, :, :] = 0

                    for j in range(new_width):
                        weight2_widened[:, j, :, :] = weight2_reshaped[:, g_mapping[j], :, :] / g_counter[g_mapping[j]]

                    # Reshape weight 2 back to the flattened form
                    new_width_fc = new_width * feature_map_side_length ** 2
                    weight2_widened = weight2_widened.view(layer2_num_output_units, new_width_fc)

                    # Update weights for layer 2
                    layer2.weight = nn.Parameter(weight2_widened)
                    layer2.in_features = new_width_fc

                else:
                    raise TypeError("Layer 2 has to be either convolutional or linear.")

                # Ensure layer2 grad's shape binding with the param data
                if layer2.weight.grad is not None:
                    if layer2.weight.grad.size() != layer2.weight.data.size():
                        layer2.weight.grad = torch.zeros_like(layer2.weight.data)

                self.wider_in_channels_visited[layername] = True

            for layername in downstream_bnorm_list:
                if self.wider_in_channels_visited[layername]:
                    continue

                batchnorm = self.node_dict[layername].layer

                # Batchnorm
                running_mean_widened = batchnorm.running_mean.clone().resize_(new_width)
                running_var_widened = batchnorm.running_var.clone().resize_(new_width)
                running_mean_widened[old_width:new_width] = 0
                running_var_widened[old_width:new_width] = 0

                for j in range(old_width, new_width):
                    running_mean_widened[j] = batchnorm.running_mean[g_mapping[j]]
                    running_var_widened[j] = batchnorm.running_var[g_mapping[j]]

                if batchnorm.affine:
                    bnorm_weight_widened = batchnorm.weight.data.clone().resize_(new_width)
                    bnorm_bias_widened = batchnorm.bias.data.clone().resize_(new_width)
                    bnorm_weight_widened[old_width:new_width] = 0
                    bnorm_bias_widened[old_width:new_width] = 0

                    for j in range(old_width, new_width):
                        bnorm_weight_widened[j] = batchnorm.weight.data[g_mapping[j]]
                        bnorm_bias_widened[j] = batchnorm.bias.data[g_mapping[j]]

                batchnorm.running_mean = running_mean_widened
                batchnorm.running_var = running_var_widened

                if batchnorm.affine:
                    batchnorm.weight = nn.Parameter(bnorm_weight_widened)
                    batchnorm.bias = nn.Parameter(bnorm_bias_widened)

                # Ensure batchnorm grad's shape binding with the param data
                if batchnorm.affine:
                    if batchnorm.weight.grad is not None:
                        if batchnorm.weight.grad.size() != batchnorm.weight.data.size():
                            batchnorm.weight.grad = torch.zeros_like(batchnorm.weight.data)

                    if batchnorm.bias.grad is not None:
                        if batchnorm.bias.grad.size() != batchnorm.bias.size():
                            batchnorm.bias.grad = torch.zeros_like(batchnorm.bias.data)

                self.wider_in_channels_visited[layername] = True

        elif "Linear" in layer1.__class__.__name__:
                assert weight1.dim() == 2

                # Widen weight1/bias1
                weight1_widened.resize_(new_width, layer1_num_input_units)
                bias1_widened.resize_(new_width)

                # Intialize the extended part (weight1) with zeros
                weight1_widened[old_width:new_width, :] = 0

                for j in range(old_width, new_width):
                    weight1_widened[j, :] = weight1[g_mapping[j], :]
                    bias1_widened[j] = bias1[g_mapping[j]]

                # Update weights for layer 1/
                layer1.weight, layer1.bias = nn.Parameter(weight1_widened), nn.Parameter(bias1_widened)
                layer1.out_features = new_width

                for layername in downstream_conv_fc_list:
                    if self.wider_in_channels_visited[layername]:
                        continue

                    layer2 = self.node_dict[layername].layer

                    # Get layer2's weight and bias
                    weight2 = layer2.weight.data

                    # Get weight2 original shape
                    layer2_num_input_units, layer2_num_output_units = weight2.size(1), weight2.size(0)

                    # Placeholders for the widened weight/bias (layer2)
                    weight2_widened = weight2.clone()

                    assert weight2.dim() == 2

                    weight2_widened.resize_(layer2_num_output_units, new_width)

                    # Intialize the extended part (weight2) with zeros
                    weight2_widened[:, old_width:new_width] = 0

                    for j in range(new_width):
                        weight2_widened[:, j] = weight2[:, g_mapping[j]] / g_counter[g_mapping[j]]

                    layer2.weight.data = nn.Parameter(weight2_widened)
                    layer2.in_features = new_width

                    # Ensure layer2 grad's shape binding with the param data
                    if layer2.weight.grad is not None:
                        if layer2.weight.grad.size() != layer2.weight.data.size():
                            layer2.weight.grad = torch.zeros_like(layer2.weight.data)

                    self.wider_in_channels_visited[layername] = True

                for layername in downstream_bnorm_list:
                    if self.wider_in_channels_visited[layername]:
                        continue

                    batchnorm = self.node_dict[layername].layer

                    # Batchnorm
                    running_mean_widened = batchnorm.running_mean.clone().resize_(new_width)
                    running_var_widened = batchnorm.running_var.clone().resize_(new_width)
                    running_mean_widened[old_width:new_width] = 0
                    running_var_widened[old_width:new_width] = 0

                    for j in range(old_width, new_width):
                        running_mean_widened[j] = batchnorm.running_mean[g_mapping[j]]
                        running_var_widened[j] = batchnorm.running_var[g_mapping[j]]

                    if batchnorm.affine:
                        bnorm_weight_widened = batchnorm.weight.data.clone().resize_(new_width)
                        bnorm_bias_widened = batchnorm.bias.data.clone().resize_(new_width)
                        bnorm_weight_widened[old_width:new_width] = 0
                        bnorm_bias_widened[old_width:new_width] = 0

                        for j in range(old_width, new_width):
                            bnorm_weight_widened[j] = batchnorm.weight.data[g_mapping[j]]
                            bnorm_bias_widened[j] = batchnorm.bias.data[g_mapping[j]]

                        batchnorm.weight = nn.Parameter(bnorm_weight_widened)
                        batchnorm.bias = nn.Parameter(bnorm_bias_widened)

                    batchnorm.running_mean = running_mean_widened
                    batchnorm.running_var = running_var_widened

                    # Ensure batchnorm grad's shape binding with the param data
                    if batchnorm.affine:
                        if batchnorm.weight.grad is not None:
                            if batchnorm.weight.grad.size() != batchnorm.weight.data.size():
                                batchnorm.weight.grad = torch.zeros_like(batchnorm.weight.data)

                        if batchnorm.bias.grad is not None:
                            if batchnorm.bias.grad.size() != batchnorm.bias.size():
                                batchnorm.bias.grad = torch.zeros_like(batchnorm.bias.data)

                    self.wider_in_channels_visited[layername] = True
        else:
            raise ValueError("Layer type not supported.")

        # Ensure layer1 grad's shape binding with the param data
        if layer1.weight.grad is not None:
            if layer1.weight.grad.size() != layer1.weight.data.size():
                layer1.weight.grad = torch.zeros_like(layer1.weight.data)

        if layer1.bias.grad is not None:
            if layer1.bias.grad.size() != layer1.bias.size():
                layer1.bias.grad = torch.zeros_like(layer1.bias.data)

        self.wider_out_channels_visited[layername1] = True

        upstream_conv_list = [self._find_upstream_layers(layername) for layername in downstream_conv_fc_list]
        upstream_conv_list = list(set([layername for layername in list(chain.from_iterable(upstream_conv_list)) if 'conv' in layername]))

        for layername in upstream_conv_list:
            _ = self.make_wider(layername, new_width, nc)

        return None

    def make_deeper(self, layername, nc, activation=None, batchnorm_flag=False, **kwargs):
        """Three conditions have to be met after each mutation:
           1. The graph internal structure has to be maintained (how nodes are connected).
           2. The node dict of the graph has to be updated so that any new generated nodes are tracked by the node dict.
           3. The layer dict of the graph has to be updated so that the optimizer can update the parameters of those nodes.
           4. Each node need to have a unique name that are tracked by both the node dict and the layer dict.
        """
        if kwargs:
            cell_num = kwargs['cell_num']
            new_name_stem = kwargs['new_layername_stem']

        layer = self.layer_dict[layername]

        if "Linear" in layer.__class__.__name__:
            # Sanity check
            assert layer.weight.dim() == 2

            new_layer = nn.Linear(layer.out_features, layer.out_features)
            # Do-nothing linear transformation
            _ = new_layer.weight.data.copy_(torch.eye(layer.out_features))
            _ = new_layer.bias.data.zero_()

            if batchnorm_flag: # Do-nothing batchnorm
                bnorm = nn.BatchNorm1d(layer.out_features)
                bnorm.weight.data.fill_(1)
                bnorm.bias.data.fill_(0)
                bnorm.running_mean.fill_(0)
                bnorm.running_var.fill_(1)

        elif "Conv" in layer.__class__.__name__:
            # Sanity check
            assert layer.kernel_size[0] % 2 == 1, "Kernel size needs to be odd"
            assert layer.weight.dim() == 4

            # Shape-preserving padding
            padding = int((layer.kernel_size[0] - 1) / 2)
            new_layer = nn.Conv2d(layer.out_channels, layer.out_channels, kernel_size=layer.kernel_size, padding=padding)

            # Do-nothing conv kernel
            _ = new_layer.weight.data.zero_()
            center_index = layer.kernel_size[0] // 2

            for i in range(0, new_layer.out_channels):
                new_layer.weight.data[i, i, center_index, center_index] = 1

            _ = new_layer.bias.data.zero_()

            if batchnorm_flag: # Do-nothing batchnorm
                bnorm = nn.BatchNorm2d(layer.out_channels)            
                bnorm.weight.data.fill_(1)
                bnorm.bias.data.fill_(0)
                bnorm.running_mean.fill_(0)
                bnorm.running_var.fill_(1)

        else:
            raise ValueError("Layer type {} not supported.".format(layer.__class__.__name__))

        old_node = self.node_dict[layername]

        if nc:
            if new_name_stem is None:
                new_name = '1_' + layername_generator('conv') if 'Conv' in new_layer.__class__.__name__ else '1_' + layername_generator('fc')
            else:
                new_name = str(cell_num) + '_' + new_name_stem
        else:
            new_name = 'nonc_' + layername_generator('conv') if 'Conv' in new_layer.__class__.__name__ else '1_' + layername_generator('fc')

        new_node = Node(new_name, new_layer, Graph.__get_op_type(new_name))

        # Adding newly created nodes to the current graph
        ## Update the node dict
        self.node_dict[new_name] = new_node

        ## Update the layer dict
        self.layer_dict[new_name] = new_node.layer

        # Update wider visited dicts
        self.wider_out_channels_visited[new_name] = False
        self.wider_in_channels_visited[new_name] = False

        # Connect the new node with the next nodes
        new_node.next_list = copy(old_node.next_list)

        # Connect the next nodes with the new node and break connection with the old
        for node in new_node.next_list:
            # Remove old node from prev_list
            node.prev_list.remove(old_node)
            node.prev_list.append(new_node)

        if activation is not None:
            if nc:
                activation_name = str(cell_num) + '_' + layername_generator(activation.__name__.lower())
            else:
                activation_name = 'nonc_' + layername_generator(activation.__name__.lower())

            activation_node = Node(activation_name, activation(inplace=True), Graph.__get_op_type(activation_name))

            old_node.next_list = [activation_node]
            activation_node.prev_list = [old_node]

            # Update the node dict and the layer dict
            self.node_dict[activation_name] = activation_node
            self.layer_dict[activation_name] = activation_node.layer

            if batchnorm_flag:
                # Create batchnorm layer
                if nc:
                    bnorm_name = str(cell_num) + '_' + layername_generator('batchnorm')
                else:
                    bnorm_name = 'nonc_' + layername_generator('batchnorm')

                bnorm_node = Node(bnorm_name, bnorm, Graph.__get_op_type(bnorm_name))

                # Connect activation layer and batchnorm layer to each other
                activation_node.next_list = [bnorm_node]
                bnorm_node.prev_list = [activation_node]

                bnorm_node.next_list = [new_node]
                new_node.prev_list = [bnorm_node]

                # Update the node dict and the layer dict
                self.node_dict[bnorm_name] = bnorm_node
                self.layer_dict[bnorm_name] = bnorm_node.layer

                # Update wider visited dicts
                self.wider_out_channels_visited[bnorm_name] = False
                self.wider_in_channels_visited[bnorm_name] = False

            else:
                # Connect activation node and new node to each other
                activation_node.next_list = [new_node]
                new_node.prev_list = [activation_node]

        else:
            if batchnorm_flag:
                if nc:
                    bnorm_name = str(cell_num) + '_' + layername_generator('batchnorm')
                else:
                    bnorm_name = 'nonc_' + layername_generator('batchnorm')

                bnorm_node = Node(bnorm_name, bnorm, Graph.__get_op_type(bnorm_name))

                # Connect batchnorm layer and old layer to each other
                old_node.next_list = [bnorm_node]
                bnorm_node.prev_list = [old_node]

                # Connect batchnorm layer and new layer to each other
                bnorm_node.next_list = [new_node]
                new_node.prev_list = [bnorm_node]

                # Update the node dict and the layer dict
                self.node_dict[bnorm_name] = bnorm_node
                self.layer_dict[bnorm_name] = bnorm_node.layer

                # Update wider visited dicts
                self.wider_out_channels_visited[bnorm_name] = False
                self.wider_in_channels_visited[bnorm_name] = False

            else:
                # Connect dropout layer and dropout layer to each other
                old_node.next_list = [new_node]
                new_node.prev_list = [old_node]

        return new_name

    def widen_kernel(self, layername, nc):
        layer = self.layer_dict[layername]

        assert layer.kernel_size[0] % 2 == 1, "Kernel size needs to be odd"
        assert layer.weight.dim() == 4, "This operation only applies to conv layers."

        old_padding = layer.padding
        padding_w, padding_h = layer.padding
        layer.padding = padding_w + 1, padding_h + 1

        out_channels, in_channels, old_kernel_w, old_kernel_h = layer.weight.data.shape
        new_kernel_w, new_kernel_h = old_kernel_w + 2, old_kernel_h + 2
        new_weight = torch.zeros((out_channels, in_channels, new_kernel_w, new_kernel_h))

        _ = new_weight[:, :, 1:(new_kernel_w - 1), 1:(new_kernel_h - 1)].copy_(layer.weight.data)

        layer.weight = nn.Parameter(new_weight)

        # Ensure layer1 grad's shape binding with the param data
        if layer.weight.grad is not None:
            if layer.weight.grad.size() != layer.weight.data.size():
                layer.weight.grad = torch.zeros_like(layer.weight.data)

        if layer.bias.grad is not None:
            if layer.bias.grad.size() != layer.bias.size():
                layer.bias.grad = torch.zeros_like(layer.bias.data)

    def add_skip_connection(self, layername, nc, activation=None, batchnorm_flag=False, **kwargs):
        """Three conditions have to be met after each mutation:
                   1. The graph internal structure has to be maintained (how nodes are connected).
                   2. The node dict of the graph has to be updated so that any new generated nodes are tracked by the node dict.
                   3. The layer dict of the graph has to be updated so that the optimizer can update the parameters of those nodes.
                   4. Each node need to have a unique name that are tracked by both the node dict and the layer dict.
        """
        if kwargs:
            cell_num = kwargs['cell_num']
            new_name_stem = kwargs['new_layername_stem']

        layer = self.layer_dict[layername]

        if "Linear" in layer.__class__.__name__:
            # Sanity check
            assert layer.weight.dim() == 2

            new_layer = nn.Linear(layer.out_features, layer.out_features)
            # Zero output weight
            _ = new_layer.weight.data.zero_()
            _ = new_layer.bias.data.zero_()

            if batchnorm_flag:  # Do-nothing batchnorm
                bnorm = nn.BatchNorm1d(layer.out_features)
                bnorm.weight.data.fill_(1)
                bnorm.bias.data.fill_(0)
                bnorm.running_mean.fill_(0)
                bnorm.running_var.fill_(1)

        elif "Conv" in layer.__class__.__name__:
            # Sanity check
            assert layer.kernel_size[0] % 2 == 1, "Kernel size needs to be odd"
            assert layer.weight.dim() == 4

            # Shape-preserving padding
            padding = int((layer.kernel_size[0] - 1) / 2)
            new_layer = nn.Conv2d(layer.out_channels, layer.out_channels,
                                  kernel_size=layer.kernel_size, padding=padding)

            # Zero-out conv kernel
            _ = new_layer.weight.data.zero_()
            _ = new_layer.bias.data.zero_()

            if batchnorm_flag:  # Do-nothing batchnorm
                bnorm = nn.BatchNorm2d(layer.out_channels)
                bnorm.weight.data.fill_(1)
                bnorm.bias.data.fill_(0)
                bnorm.running_mean.fill_(0)
                bnorm.running_var.fill_(1)

        else:
            raise ValueError("Layer type {} not supported.".format(layer.__class__.__name__))

        old_node = self.node_dict[layername]

        if nc:
            if new_name_stem is None:
                new_name = '1_' + layername_generator('conv')
            else:
                new_name = str(cell_num) + '_' + new_name_stem
        else:
            new_name = 'nonc_' + layername_generator('conv')

        new_node = Node(new_name, new_layer, Graph.__get_op_type(new_name))

        # Adding newly created nodes to the current graph
        ## Update the node dict
        self.node_dict[new_name] = new_node

        ## Update the layer dict
        self.layer_dict[new_name] = new_node.layer

        # Update wider visited dicts
        self.wider_out_channels_visited[new_name] = False
        self.wider_in_channels_visited[new_name] = False

        # Connect both the old node and the new node with the next nodes
        new_node.next_list = copy(old_node.next_list)

        # Connect the next nodes with the new node while keeping the connection with the old (do nothing)
        for node in new_node.next_list:
            node.prev_list.append(new_node)

        if activation is not None:
            # Connect old node to the new activation and vice versa
            if nc:
                activation_name = str(cell_num) + '_' + layername_generator(activation.__name__.lower())
            else:
                activation_name = 'nonc_' + layername_generator(activation.__name__.lower())

            activation_node = Node(activation_name, activation(inplace=True), Graph.__get_op_type(activation_name))

            old_node.next_list.append(activation_node)
            activation_node.prev_list = [old_node]

            # Update the node dict and the layer dict
            self.node_dict[activation_name] = activation_node
            self.layer_dict[activation_name] = activation_node.layer

            if batchnorm_flag:
                # Create batchnorm layer
                if nc:
                    bnorm_name = str(cell_num) + '_' + layername_generator('batchnorm')
                else:
                    bnorm_name = 'nonc_' + layername_generator('batchnorm')

                bnorm_node = Node(bnorm_name, bnorm, Graph.__get_op_type(bnorm_name))

                # Connect activation layer and batchnorm layer to each other
                activation_node.next_list = [bnorm_node]
                bnorm_node.prev_list = [activation_node]

                bnorm_node.next_list = [new_node]
                new_node.prev_list = [bnorm_node]

                # Update the node dict and the layer dict
                self.node_dict[bnorm_name] = bnorm_node
                self.layer_dict[bnorm_name] = bnorm_node.layer

                # Update wider visited dicts
                self.wider_out_channels_visited[bnorm_name] = False
                self.wider_in_channels_visited[bnorm_name] = False

            else:
                # Connect dropout node and new node to each other
                activation_node.next_list = [new_node]
                new_node.prev_list = [activation_node]
        else:
            if batchnorm_flag:
                if nc:
                    bnorm_name = str(cell_num) + '_' + layername_generator('batchnorm')
                else:
                    bnorm_name = 'nonc_' + layername_generator('batchnorm')

                bnorm_node = Node(bnorm_name, bnorm, Graph.__get_op_type(bnorm_name))

                # Connect batchnorm layer and old layer to each other
                old_node.next_list.append(bnorm_node)
                bnorm_node.prev_list = [old_node]

                # Connect dropout layer and new layer to each other
                bnorm_node.next_list = [new_node]
                new_node.prev_list = [bnorm_node]

                self.node_dict[bnorm_name] = bnorm_node
                self.layer_dict[bnorm_name] = bnorm_node.layer

                # Update wider visited dicts
                self.wider_out_channels_visited[bnorm_name] = False
                self.wider_in_channels_visited[bnorm_name] = False

            else:
                # Connect dropout layer and dropout layer to each other
                old_node.next_list.append(new_node)
                new_node.prev_list = [old_node]

        return new_name

    @staticmethod
    def __get_op_type(name):
        supported_op_types = {'conv', 'fc', 'maxpool', 'batchnorm', 'dropout', 
                              'add', 'cat', 'flatten', 'relu', 'elu', 'logsoftmax'}

        op_type = name.split('_')[1]

        if op_type not in supported_op_types:
            raise ValueError("Op type {} not supported.".format(name))

        return op_type


class Flatten(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.view(-1, x.size(1) * x.size(2) * x.size(3))


def concat(output_tuple):
    concated_output = torch.cat(output_tuple, dim=1)

    return concated_output


def layername_generator(op_type):
    time_digits = str(time.time() % 1)[-4:]
    rand_digits = ''.join([str(random.randint(0, 9)) for _ in range(4)])

    return op_type + '_' + time_digits + rand_digits


def get_layername_stem(layername):
    return '_'.join(layername.split('_')[1:])



