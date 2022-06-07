"""
This file contains the SNN simulation classes and the Neuroevolution (via NEAT) of SNN's.

The data is being loaded in this file in order to test GPU usage
"""

# Auxiliar
import torch
import numpy as np
from random import random, choice
import pickle
import time

from Scripts import visualize

# BindsNET
from bindsnet.network import Network
from bindsnet.network.nodes import Input, LIFNodes
from bindsnet.network.topology import Connection
from bindsnet.network.monitors import Monitor
from bindsnet.analysis.plotting import plot_spikes, plot_voltages, plot_performance, plot_weights
from bindsnet.evaluation.evaluation import all_activity

# NEAT
import neat
from neat.attributes import FloatAttribute, BoolAttribute, StringAttribute
from neat.config import ConfigParameter, write_pretty_params
from neat.genes import BaseGene, DefaultConnectionGene
from neat.genome import DefaultGenomeConfig
from neat.graphs import creates_cycle, required_for_output
from neat.six_util import iteritems, iterkeys
from itertools import count

device = 'cpu'  # cuda or cpu
number = 2
delimit = True
limit = 5

if not delimit:
    limit = 168

# Test Data
test_label = torch.load(f'Files/Sets/Set{number}/TEST/Test_labels.pt', map_location=torch.device(device))
test_data = torch.load(f'Files/Sets/Set{number}/TEST/Test_data_encoded.pt', map_location=torch.device(device))
test_times = torch.load(f'Files/Sets/Set{number}/TEST/Test_times.pt', map_location=torch.device(device))
vMin_test = -torch.min(test_data)
test_data = test_data + vMin_test
vMax_test = torch.max(test_data)
test_data = test_data / vMax_test

# Train Data
train_label = torch.load(f'Files/Sets/Set{number}/TRAIN/Train_labels.pt', map_location=torch.device(device))
train_data = torch.load(f'Files/Sets/Set{number}/TRAIN/Train_data_encoded.pt', map_location=torch.device(device))
train_times = torch.load(f'Files/Sets/Set{number}/TRAIN/Train_times.pt', map_location=torch.device(device))
if delimit:
    idx = torch.randperm(train_label.shape[0])[:limit]

    train_label = train_label[idx]
    train_data = train_data[idx]
    train_times = train_times[idx]

vMin_train = -torch.min(train_data)
train_data = train_data + vMin_train
vMax_train = torch.max(train_data)
train_data = train_data / vMax_train

print('Data loaded: ')
print(f'Train: Max = {vMax_train} - Min = {vMin_train} - Duration = {train_data.shape}')
print(f'Test: Max = {vMax_test} - Min = {vMin_test} - Duration = {test_data.shape}')


# ---------------------------------------- SNN implementation -------------------------------------------------------- #
class Net:
    """
        Spiking Neural Networks
    """

    def __init__(self, in_nodes, out_nodes, hidden_nodes, connections, device):
        """
        Initialization of a SNN

        :param list in_nodes: ID of input nodes (Sorted).
        :param list out_nodes: ID of output nodes (Sorted).
        :param list hidden_nodes: ID of hidden nodes (Sorted).
        :param dict connections: Dict of weights (Only enabled) Ej. {(in, out): value}.
        :param str device: Which device will handling the net.
        """

        # Variables of the Net
        self.in_nodes = in_nodes
        self.out_nodes = out_nodes
        self.hidden_nodes = hidden_nodes
        self.connections = connections

        # Auxiliary variables
        self.device = device
        if device == 'cuda':
            self.gpu = True
        else:
            self.gpu = False

        # Tensors fo the Net
        self.inputs = torch.tensor(self.in_nodes, device=self.device, dtype=torch.float32)
        self.outputs = torch.tensor(self.out_nodes, device=self.device, dtype=torch.float32)
        self.hidden = torch.tensor(self.hidden_nodes, device=self.device, dtype=torch.float32)

        self.num_input = self.inputs.shape[0]
        self.num_output = self.outputs.shape[0]
        self.num_hidden = self.hidden.shape[0]

        # Network construction
        # ------> Net initialization
        self.network = Network()
        # ------> Nodes
        self.input_l = Input(n=self.num_input)
        self.output_l = LIFNodes(n=self.num_output, tc_decay=1)  # TODO: Check the tc_decay property
        # TODO: Consider more hidden layers
        if self.num_hidden > 0:
            self.hidden_l = LIFNodes(n=self.num_hidden)
        else:
            self.hidden_l = None

        # ------> Monitors
        self.monitorIn = Monitor(self.input_l, state_vars=("s",), device=self.device)
        self.network.add_monitor(monitor=self.monitorIn, name="Inputs")
        self.monitorOut = Monitor(self.output_l, state_vars=("s", "v"), device=self.device)
        self.network.add_monitor(monitor=self.monitorOut, name="Outputs")

        if self.num_hidden > 0:
            self.monitorHid = Monitor(self.hidden_l, state_vars=("s", "v"), device=self.device)
            self.network.add_monitor(monitor=self.monitorHid, name="Hidden")

        # ------> Adding Layers
        self.network.add_layer(layer=self.input_l, name="Input")
        if self.hidden_l:
            self.network.add_layer(layer=self.hidden_l, name="Hidden")
        self.network.add_layer(layer=self.output_l, name="Output")

        # TODO: Connections are created even if the connection is 0.
        # ------> Adding Connections
        #           Input - Output
        self.create_conn(in_layer=self.input_l, in_nodes=self.inputs, in_name="Input",
                         out_layer=self.output_l, out_nodes=self.outputs, out_name="Output")
        if self.hidden_l:
            #        Input - Hidden
            self.create_conn(in_layer=self.input_l, in_nodes=self.inputs, in_name="Input",
                             out_layer=self.hidden_l, out_nodes=self.hidden, out_name="Hidden")
            #        Hidden - Hidden (Recurrent)
            self.create_conn(in_layer=self.hidden_l, in_nodes=self.hidden, in_name="Hidden",
                             out_layer=self.hidden_l, out_nodes=self.hidden, out_name="Hidden")
            #        Hidden - Output
            self.create_conn(in_layer=self.hidden_l, in_nodes=self.hidden, in_name="Hidden",
                             out_layer=self.output_l, out_nodes=self.outputs, out_name="Output")

        if self.gpu:
            self.network.cuda()

    def create_conn(self, in_layer, in_nodes, in_name, out_layer, out_nodes, out_name):
        """
        Adds connections between two layers to the Network

        :param LIFNodes or Input in_layer: Instance of BindNet Layer
        :param torch.Tensor or list in_nodes: List of In Nodes
        :param str in_name: Name of the In Layer
        :param LIFNodes out_layer: Instance of BindNet Layer
        :param torch.Tensor or list out_nodes: List of Out Nodes
        :param str out_name: Name of the Out Layer
        :return: None
        """
        weights = self.create_w(in_nodes, out_nodes)
        current_conn = Connection(source=in_layer,
                                  target=out_layer,
                                  w=weights)
        self.network.add_connection(connection=current_conn,
                                    source=in_name,
                                    target=out_name)

    def create_w(self, in_nodes, out_nodes):
        """
        Creates a Torch Tensor of the weights between two layers
        :param torch.Tensor or list in_nodes: Sorted array of input nodes.
        :param torch.Tensor or list out_nodes: Sorted array of output nodes.
        :return: Tensor of weights (Torch Object)
        """
        # Get dimensiones
        num_i = in_nodes.shape[0]
        num_o = out_nodes.shape[0]

        # Create zero's Tensor
        weights = torch.zeros([num_i, num_o], device=self.device, dtype=torch.float32)

        # Iterate over in_nodes
        for k, id_i in enumerate(in_nodes):
            # Iterate over out_nodes
            for o, id_o in enumerate(out_nodes):
                w = self.connections.get((int(id_i), int(id_o)))
                if w:
                    weights[k, o] = w
        return weights

    def report(self):
        return f'In Nodes: {self.in_nodes} - Out Nodes: {self.out_nodes} - ' \
               f'Hidden Nodes: {self.hidden_nodes} - Connections: {self.connections}'

    def train(self, one_step=True, reset=True):
        """
        Run a simulation of the SNN with the given data
        Former 'Simulation def'

        :param bool one_step: Whether to perform in one-step.
        :param bool reset: Whether to reset variables.
        :return: Accuracy, time of first Spike, mean of SHP
        """

        # Number of examples
        nData = train_data.shape[0]

        # Control
        result = False
        nTime = 0

        # Evaluation
        counter = 0
        tp, fp, fn, tn = 0, 0, 0, 0
        _sph = []
        avgTrigger = []
        hiddenSpikes = []
        stme = time.time()
        for nStep, (batch, label, true_time) in enumerate(zip(train_data, train_label, train_times)):
            # print(f'Data {nStep + 1}.')

            for nTime, spikes in enumerate(batch):
                self.network.run({"Input": spikes}, time=1, one_step=one_step)
                result = self.monitorOut.get("s")[0, 0, 0]
                if self.num_hidden > 0:
                    hiddenSpikes.append(self.monitorHid.get('s'))
                if result:
                    _sph.append(int(true_time - nTime))
                    avgTrigger.append(nTime)
                    break

            if reset:
                self.network.reset_state_variables()

            # print(f'Expected: {label} - Result: {result}')
            # counter += 1 if label == result else 0
            if label == 1 and result == 1:
                tp += 1
                counter += 1
            elif label == 0 and result == 0:
                tn += 1
                counter += 1
            elif label == 0 and result == 1:
                fp += 1
            elif label == 1 and result == 0:
                fn += 1

        t = time.time() - stme
        sph = counter * np.mean(_sph)
        return counter / nData, (len(avgTrigger), np.mean(avgTrigger)), \
               {'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn, 'SPH': sph}, t, hiddenSpikes

    def test(self, one_step=True, reset=True):
        """
        Run a simulation of the SNN with the given data
        Former 'Simulation def'

        :param bool one_step: Whether to perform in one-step.
        :param bool reset: Whether to reset variables.
        :return: Accuracy, time of first Spike, mean of SHP
        """

        # Number of examples
        nData = train_data.shape[0]

        # Control
        result = False
        nTime = 0

        # Evaluation
        counter = 0
        tp, fp, fn, tn = 0, 0, 0, 0
        sph = []
        avgTrigger = []
        hiddenSpikes = []

        for nStep, (batch, label, true_time) in enumerate(zip(test_data, test_label, test_times)):
            # print(f'Data {nStep + 1}.')
            for nTime, spikes in enumerate(batch):
                self.network.run({"Input": spikes}, time=1, one_step=one_step)
                result = self.monitorOut.get("s")[0, 0, 0]
                if self.num_hidden > 0:
                    hiddenSpikes.append(self.monitorHid.get('s'))

                if result:
                    sph.append(int(true_time - nTime))
                    avgTrigger.append(nTime)
                    break

            if reset:
                self.network.reset_state_variables()

            # print(f'Expected: {label} - Result: {result}')
            # counter += 1 if label == result else 0
            if label == 1 and result == 1:
                tp += 1
                counter += 1
            elif label == 0 and result == 0:
                tn += 1
                counter += 1
            elif label == 0 and result == 1:
                fp += 1
            elif label == 1 and result == 0:
                fn += 1

        return counter / nData, (len(avgTrigger), np.mean(avgTrigger)), \
               {'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn, 'SPH': sph}, hiddenSpikes

    def play(self, data, labels, one_step=True, reset=True):
        """
        Run a simulation of the SNN with the given data

        :param list or np.ndarray or torch.Tensor data: List of Test data.
        :param list labels: List of True labels.
        :param bool one_step: Whether to perform in one-step.
        :param bool reset: Whether to reset variables.
        :return: Accuracy, time of first Spike, mean of SHP
        """
        # Preparation
        # torchData = data
        simTime = data.shape[1]
        # torchLabels = torch.tensor(labels, device=self.device, dtype=torch.float32)
        # Number of examples
        nData = data.shape[0]

        # Control
        result = False
        # nTime = 0

        # evaluation
        counter = 0
        tp, fp, fn, tn = 0, 0, 0, 0
        avgTrigger = []

        for nStep, (batch, label) in enumerate(zip(data, labels)):
            # print(f'Data {nStep + 1}.')
            for nTime, spikes in enumerate(batch):
                self.network.run({"Input": spikes}, time=1, one_step=one_step)
                result = self.monitorOut.get("s")[0, 0, 0]
                if result:
                    avgTrigger.append(nTime)
                    break
                # volt = self.monitorOut.get('v')
                # print(volt)
            if reset:
                self.network.reset_state_variables()

            # print(f'Expected: {label} - Result: {result}')
            # counter += 1 if label == result else 0
            if label == 1 and result == 1:
                tp += 1
                counter += 1
            elif label == 0 and result == 0:
                tn += 1
                counter += 1
            elif label == 0 and result == 1:
                fp += 1
            elif label == 1 and result == 0:
                fn += 1

        return counter / nData, (len(avgTrigger), np.mean(avgTrigger)), {'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn}

    def XOR(self):
        c = 10
        inZero = np.zeros(c, dtype=int)
        inOne = np.zeros(c, dtype=int)
        # inBias = np.zeros(c, dtype=int)
        # inZero[1] = 1
        inOne[2] = 1

        # in1 = [inBias, inBias, inBias, inBias]
        in1 = [inZero, inZero, inOne, inOne]
        in2 = [inZero, inOne, inZero, inOne]

        train_data = torch.tensor(np.stack((in1, in2), axis=-1), dtype=torch.float32, device=device)
        # Single output node
        train_labels = [False, True, True, False]
        # Twu output node
        # train_labels = [[True, False], [False, True], [False, True], [True, False]]
        score = self.play(train_data, train_labels, one_step=True, reset=True)

        return score

    def AND(self):
        c = 10
        inZero = np.zeros(c, dtype=int)
        inOne = np.zeros(c, dtype=int)

        inOne[2] = 1

        in1 = [inZero, inZero, inOne, inOne]
        in2 = [inZero, inOne, inZero, inOne]

        train_data = torch.tensor(np.stack((in1, in2), axis=-1), dtype=torch.float32, device=device)

        train_labels = [False, False, False, True]
        score = self.play(train_data, train_labels)
        return score


# ---------------------------------------- NEAT implementation ------------------------------------------------------- #
class LIFNodeGene(BaseGene):
    """
    Contains attributes for the iznn node genes and determines genomic distances. NOT IMPLEMENTED YET!!
    To implement a parameter search, check the Izhikevich implementation (NEAT doc)
    """
    __gene_attributes__ = []

    def distance(self, other, config):
        """
        Measures differences between nodes. NOT IMPLEMENTED YET!!
        :param other:
        :param config:
        :return:
        """

        return 0


class LIFConnectionGene(BaseGene):
    __gene_attributes__ = [StringAttribute('component'),
                           FloatAttribute('values'),
                           BoolAttribute('enabled')]

    def distance(self, other, config):
        """
        Groups attributes specific to connection genes - such as weight - and calculates genetic distances between
        two homologous (not disjoint or excess) connection genes.

        :param LIFConnectionGene other:
        :param LIFConnectionGene config:
        :return: Genetic Distance
        """

        d = np.abs(self.values - other.values)
        if self.enabled == other.enabled:
            d += 1
        return d * config.compatibility_weight_coefficient


class LIFGenomeConfig(object):
    __params = [ConfigParameter('num_inputs', int),
                ConfigParameter('num_outputs', int),
                ConfigParameter('num_hidden', int),
                ConfigParameter('compatibility_disjoint_coefficient', float),
                ConfigParameter('compatibility_weight_coefficient', float),
                ConfigParameter('conn_add_prob', float),
                ConfigParameter('conn_delete_prob', float),
                ConfigParameter('node_add_prob', float),
                ConfigParameter('node_delete_prob', float),
                ConfigParameter('feed_forward', bool),
                ConfigParameter('single_structural_mutation', bool),
                ConfigParameter('structural_mutation_surer', bool),
                ConfigParameter('initial_connection', str)]

    def __init__(self, params):
        # Ignoring set of available activation functions.
        self.__params += LIFNodeGene.get_config_params()
        self.__params += LIFConnectionGene.get_config_params()

        for p in self.__params:
            setattr(self, p.name, p.interpret(params))

        # By design, Input Nodes are 0, ..., N, and Output Nodes are N, ..., M
        self.input_keys = [i for i in range(self.num_inputs)]
        self.output_keys = [i for i in range(self.num_inputs, self.num_inputs + self.num_outputs)]

        self.node_indexer = None

    def save(self, f):
        write_pretty_params(f, self, self.__params)

    def get_new_node_key(self, node_dict):
        if self.node_indexer is None:
            self.node_indexer = count(max(list(iterkeys(node_dict))) + 1)

        new_id = next(self.node_indexer)

        assert new_id not in node_dict
        return new_id


class LIFGenome(object):
    @classmethod
    def parse_config(cls, param_dict):
        return LIFGenomeConfig(param_dict)

    @classmethod
    def write_config(cls, f, config):
        config.save(f)

    def __init__(self, key):
        self.key = key

        # (gene_key, gene) pair for gene sets
        self.connections = {}
        self.nodes = {}

        # Fitness results
        self.fitness = None

        # Other results
        self.others = None
        self.trigger = None
        self.time = None
        self.spikes = None

    def configure_new(self, config):
        # Create node genes for the output nodes.
        for node_key in config.output_keys:
            self.nodes[node_key] = self.create_node(config, node_key)

        # Add connections based on initial connectivity
        if config.initial_connection == 'Fully':
            self.connect_full_direct(config)
        elif config.initial_connection == 'FS':
            self.connect_fs_neat(config)
        # for input_id in config.input_keys:
        #     for node_id in iterkeys(self.nodes):
        #         connection = self.create_connection(config, input_id, node_id)
        #         self.connections[connection.key] = connection

    def get_new_node_key(self, config):
        """ Compute the corespondent new ID of Node """
        # TODO: Check if holds
        print('In Self get new node key')
        new_id = 19
        while new_id in self.nodes:
            new_id += 1

        return new_id

    def configure_crossover(self, genome1, genome2, config):
        """ Configure a new genome by crossover from two parent genomes """

        # Select parent 1 and 2 by their fitness
        if genome1.fitness > genome2.fitness:
            parent1, parent2 = genome1, genome2
        else:
            parent1, parent2 = genome2, genome1

        # Inherit connection genes
        for key, cg1 in iteritems(parent1.connections):
            cg2 = parent2.connections.get(key)
            if cg2 is None:
                # Meaning that parent 2 doesn't have that connection
                # So, copy the connection from fittest parent.
                self.connections[key] = cg1.copy()
            else:
                # Both parents have the connection, so combine genes
                self.connections[key] = cg1.crossover(cg2)

        # Inherit node genes
        parent1_set = parent1.nodes
        parent2_set = parent2.nodes

        for key, ng1 in iteritems(parent1_set):
            ng2 = parent2_set.get(key)
            # Verify if current node is not in nodes
            assert key not in self.nodes

            if ng2 is None:
                # Meaning that parent 2 doesn't have that node
                # So, copy the node from fittest parent.
                self.nodes[key] = ng1.copy()
            else:
                # Both parents have the node, so combine genes
                self.nodes[key] = ng1.crossover(ng2)

    def mutate(self, config):
        """ Mutation of LIF Genome """

        if config.single_structural_mutation:
            div = max(1, (config.node_add_prob + config.node_delete_prob +
                          config.conn_add_prob + config.conn_delete_prob))
            r = random()
            if r < (config.node_add_prob / div):
                self.mutate_add_node(config)
            elif r < ((config.node_add_prob + config.node_delete_prob) / div):
                self.mutate_delete_node(config)
            elif r < ((config.node_add_prob + config.node_delete_prob +
                       config.conn_add_prob) / div):
                self.mutate_add_connection(config)
            elif r < ((config.node_add_prob + config.node_delete_prob +
                       config.conn_add_prob + config.conn_delete_prob) / div):
                self.mutate_delete_connection()
        else:
            # Add node
            if random() < config.node_add_prob:
                self.mutate_add_node(config)
            # Delete node
            if random() < config.node_delete_prob:
                self.mutate_delete_node(config)
            # Add connection
            if random() < config.conn_add_prob:
                self.mutate_add_connection(config)
            # Delete connection
            if random() < config.conn_delete_prob:
                self.mutate_delete_connection()

        # Mutate connection genes
        for cg in self.connections.values():
            cg.mutate(config)

        # Mutate node genes (bias, response, etc.)
        for ng in self.nodes.values():
            ng.mutate(config)

    def mutate_add_node(self, config):
        if not self.connections:
            if config.structural_mutation_surer:
                self.mutate_add_connection(config)
            # Not connections available
            # return None, None
            return

        # Choose a random connection to split
        conn_to_split = choice(list(self.connections.values()))

        # Create new node
        # new_node_id = self.get_new_node_key()
        # TODO: Check if holds
        new_node_id = config.get_new_node_key(self.nodes)
        ng = self.create_node(config, new_node_id)
        self.nodes[new_node_id] = ng

        # Disable this connection and create two new connections joining its nodes via
        # the given node.  The new node+connections have roughly the same behavior as
        # the original connection (depending on the activation function of the new node).
        conn_to_split.enabled = False
        i, o = conn_to_split.key
        self.add_connection(config, i, new_node_id, 1.0, True)
        self.add_connection(config, new_node_id, o, conn_to_split.values, True)

    def add_connection(self, config, input_key, output_key, weight, enabled):
        assert isinstance(input_key, int)
        assert isinstance(output_key, int)
        assert output_key >= 0
        assert isinstance(enabled, bool)

        key = (input_key, output_key)
        connection = LIFConnectionGene(key)
        connection.init_attributes(config)
        connection.values = weight
        connection.enabled = enabled
        self.connections[key] = connection

    def mutate_add_connection(self, config):
        """ Attempt to add a new connection. The only restriction being that the output
            node cannot be one of the network input nodes."""

        possible_outputs = list(iterkeys(self.nodes))
        out_node = choice(possible_outputs)

        possible_inputs = possible_outputs + config.input_keys
        in_node = choice(possible_inputs)

        # Don't duplicate nodes. If duplicated, toggled to enabled
        key = (in_node, out_node)
        if key in self.connections:
            if config.structural_mutation_surer:
                self.connections[key].enabled = True
            return

        # Don't allow connections between two output nodes
        if in_node in config.output_keys and out_node in config.output_keys:
            return

        # For feed-forward networks, avoid creating cycles
        if config.feed_forward and creates_cycle(list(iterkeys(self.connections)), (in_node, out_node)):
            return

        # Don't allow recurrent connections if feedforward
        if config.feed_forward and in_node == out_node:
            return

        cg = self.create_connection(config, in_node, out_node)
        self.connections[cg.key] = cg

    def mutate_delete_node(self, config):
        available_nodes = [(k, v) for k, v in iteritems(self.nodes) if k not in config.output_keys]

        # Do nothing if there are no available nodes
        if not available_nodes:
            return -1

        del_key, _ = choice(available_nodes)

        # Set all connections of/from node selected
        # OLD
        # connections_to_delete = set()
        # for k, v in iteritems(self.connections):
        #     if del_key in v.key:
        #         connections_to_delete.add(v.key)
        connections_to_delete = [v.key for _, v in iteritems(self.connections) if del_key in v.key]

        for key in connections_to_delete:
            del self.connections[key]

        del self.nodes[del_key]

        return del_key

    def mutate_delete_connection(self):
        if self.connections:
            key = choice(list(self.connections.keys()))
            del self.connections[key]

    def distance(self, other, config):
        """
        Returns the genetic distance between this genome and the other. This distance value
        is used to compute genome compatibility for speciation.
        :param other: Node to compare with.
        :param config: Config params.
        :return: genetic distance
        """
        # Compute node gene distance component
        node_distance = 0.0
        if self.nodes or other.nodes:
            disjoint_nodes = 0
            for k2 in iterkeys(other.nodes):
                if k2 not in self.nodes:
                    disjoint_nodes += 1

            for k1, n1 in iteritems(self.nodes):
                n2 = other.nodes.get(k1)
                if n2 is None:
                    disjoint_nodes += 1
                else:
                    # Homologous genes compute their own distance value
                    node_distance += n1.distance(n2, config)

            max_nodes = max(len(self.nodes), len(other.nodes))
            node_distance = (node_distance + config.compatibility_disjoint_coefficient * disjoint_nodes) / max_nodes

        # Compute connection gene differences
        conn_distance = 0.0
        if self.connections or other.connections:
            disjoint_conn = 0
            for k2 in iterkeys(other.connections):
                if k2 not in self.connections:
                    disjoint_conn += 1

            for k1, c1 in iteritems(self.connections):
                c2 = other.connections.get(k1)
                if c2 is None:
                    disjoint_conn += 1
                else:
                    # Homologous genes compute their own distance value
                    conn_distance += c1.distance(c2, config)

            max_conn = max(len(self.connections), len(other.connections))
            conn_distance = (conn_distance + config.compatibility_disjoint_coefficient + disjoint_conn) / max_conn

        distance = node_distance + conn_distance
        return distance

    def size(self):
        """ Returns genomes complexity, taken to be (number of nodes, number of enabled connections)"""
        num_enables_conn = sum([1 for cg in self.connections.values() if cg.enabled])
        return len(self.nodes), num_enables_conn

    def __str__(self):
        s = 'Nodes'
        for k, ng in iteritems(self.nodes):
            s += f'\n\t{k} {ng}'

        s += '\nConnections:'
        connections = list(self.connections.values())
        connections.sort()
        for c in connections:
            s += f'\n\t{c}'

        return s

    def add_hidden_nodes(self, config):
        for i in range(config.num_hidden):
            node_key = self.get_new_node_key()
            # Verify if current node is not in nodes
            assert node_key not in self.nodes
            node = self.__class__.create_node(config, node_key)
            self.nodes[node_key] = node

    @staticmethod
    def create_node(config, node_id):
        node = LIFNodeGene(node_id)
        node.init_attributes(config)
        return node

    @staticmethod
    def create_connection(config, input_id, output_id):
        connection = LIFConnectionGene((input_id, output_id))
        connection.init_attributes(config)
        return connection

    def connect_fs_neat(self, config):
        """ Randomly connect one input to all output nodes (FS-NEAT) """
        input_id = choice(config.input_keys)
        for output_id in config.output_keys:
            connection = self.create_connection(config, input_id, output_id)
            self.connections[connection.key] = connection

    def connect_full_direct(self, config):
        """ Create a fully-connected genome, including direct input-output connections. """
        for input_id in config.input_keys:
            for node_id in iterkeys(self.nodes):
                connection = self.create_connection(config, input_id, node_id)
                self.connections[connection.key] = connection


def build_hidden_layers(inputs, outputs, connections):
    """
    Collect the layers whose members can be evaluated in parallel in a feed-forward network.
    This is a modification of feed_forward_layers in neat.graphs

    :param inputs: list of the network input nodes
    :param outputs: list of the output node identifiers
    :param connections: list of (input, output) connections in the network.

    Returns a list of layers, with each layer consisting of a set of node identifiers.
    Note that the returned layers do not contain nodes whose output is ultimately
    never used to compute the final network output.
    """
    required = required_for_output(inputs, outputs, connections)
    layers = []
    set_inputs = set(inputs)
    set_outputs = set(outputs)
    nodes = []
    while True:
        # Find candidate nodes c for the next layer.  These nodes should connect
        # a node in s to a node not in s.
        candidates = set(
            b for (a, b) in connections if a in set_inputs and b not in set_inputs and b not in set_outputs)
        # Keep only the used nodes whose entire input set is contained in s.
        current_layer = set()
        for node in candidates:
            if node in required and all(i in set_inputs for (i, o) in connections if o == node):
                current_layer.add(node)
                nodes.append(node)

        if not current_layer:
            break

        layers.append(current_layer)
        set_inputs = set_inputs.union(current_layer)

    return layers, nodes


def simulate(genome, config, simple=True):
    in_nodes = config.genome_config.input_keys
    out_nodes = config.genome_config.output_keys
    connections = {k: v.values for k, v in genome.connections.items() if v.enabled}
    _, hidden_nodes = build_hidden_layers(inputs=in_nodes,
                                          outputs=out_nodes,
                                          connections=connections)

    net = Net(in_nodes=in_nodes, out_nodes=out_nodes,
              hidden_nodes=hidden_nodes, connections=connections, device=device)

    accuracy, trigger, moreR, t, spikes = net.train()

    sens = moreR['TP'] / (moreR['TP'] + moreR['FN'])
    spec = moreR['TN'] / (moreR['TN'] + moreR['FP'])

    # Simple means that the score is the accuracy. If false, the score will be set as the
    #   Accuracy * Average SPH
    if simple:
        score = accuracy
    else:
        score = accuracy * np.mean(moreR['SPH'])
    print(f'Genome {genome.key} -> Score: {score} - Acc: {accuracy} - Sens: {sens} - Spec: {spec} - TP: {moreR["TP"]} -'
          f' FP: {moreR["FP"]}- FN: {moreR["FN"]} - TN: {moreR["TN"]} - Trig-Count: {trigger[0]} - '
          f'Trig-Avg: {trigger[1]} - SPHs: {moreR["SPH"]}')

    return score, moreR, trigger, t, spikes


def eval_genome(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness, genome.others, genome.trigger, genome.time, genome.spikes = simulate(genome, config,
                                                                                             simple=True)


def run(num_test):
    print('Running')
    conf_file = 'config'
    conf_test = neat.Config(LIFGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet,
                            neat.DefaultStagnation, conf_file)

    population = neat.Population(conf_test)
    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)
    population.run(eval_genome, 400)
    print(population.best_genome)

    visualize.draw_net(conf_test, population.best_genome, filename=f'Results/Net_{num_test}')
    visualize.plot_stats(stats, ylog=False, view=True, filename=f'Results/avg_fitness{num_test}.svg')
    visualize.plot_species(stats, view=True, filename=f'Results/speciation{num_test}.svg')

    return population.best_genome, population


if __name__ == '__main__':
    num = 1

    stime = time.time()
    best, population_ = run(num)
    print(f'Total time of execution: {time.time() - stime}')

    # Saving best SNN
    i_nodes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
    o_nodes = [18]
    conn = {k: v.values for k, v in best.connections.items() if v.enabled}
    _, h_nodes = build_hidden_layers(i_nodes, o_nodes, conn)
    snn_info = [h_nodes, conn]
    # num = input('Save number: ')

    with open(f'SavedSNNs/SNN_{num}.pkl', 'wb') as file:
        pickle.dump(snn_info, file)
    #
    print('\n------------Testing------------\n')

    snn = Net(in_nodes=i_nodes, out_nodes=o_nodes, hidden_nodes=h_nodes,
              connections=conn, device=device)
    test_results = snn.test()
    print(test_results)
