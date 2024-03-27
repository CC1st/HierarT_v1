import os
import networkx as nx
from collections import defaultdict
import numpy as np
import torch
import math


class Env(object):
    def __init__(self, examples, config, state_action_space=None):
        """Temporal Knowledge Graph Environment.
        examples: quadruples (subject, relation, object, timestamps);
        config: config dict;
        state_action_space: Pre-processed action space;
        """
        self.config = config
        self.num_rel = config['num_rel']
        self.graph, self.label2nodes = self.build_graph(examples)
        # [0, num_rel) -> normal relations; num_rel -> stay in place，(num_rel, num_rel * 2] reversed relations.
        self.NO_OP = self.num_rel  # Stay in place; No Operation
        self.ePAD = config['num_ent']  # Padding entity
        self.rPAD = config['num_rel'] * 2 + 1  # Padding relation.
        self.tPAD = 0  # Padding time
        self.state_action_space = state_action_space

        if self.state_action_space:
            self.state_action_space_key = self.state_action_space.keys()

        self.entities_list = []

        self.dataset = config['dataset']


    def build_graph(self, examples):
        """The graph node is represented as (entity, time), and the edges are directed and labeled relation.
        return:
            graph: nx.MultiDiGraph;
            label2nodes: a dict [keys -> entities, value-> nodes in the graph (entity, time)]
        """
        graph = nx.MultiDiGraph()
        label2nodes = defaultdict(set)
        examples.sort(key=lambda x: x[3], reverse=True)  # Reverse chronological order
        for example in examples:
            src = example[0]
            rel = example[1]
            dst = example[2]
            time = example[3]

            # Add the nodes and edges of the current quadruple
            src_node = (src, time)
            dst_node = (dst, time)
            if src_node not in label2nodes[src]:
                graph.add_node(src_node, label=src)
            if dst_node not in label2nodes[dst]:
                graph.add_node(dst_node, label=dst)

            graph.add_edge(src_node, dst_node, relation=rel)
            graph.add_edge(dst_node, src_node, relation=rel+self.num_rel+1)

            label2nodes[src].add(src_node)
            label2nodes[dst].add(dst_node)
        return graph, label2nodes

    def get_state_actions_space_complete_predata(self, entity, time, current_=False, max_option_num=None, max_action_num=None):
        """Get the action space of the current state.
        """
        # if max_action_num is None:
        #     max_action_num = self.config['max_action_num']
        #     max_option_num = self.config['max_option_num']
        if self.state_action_space:
            if (entity, time, current_) in self.state_action_space_key:
                return self.state_action_space[(entity, time, current_)]
        nodes = self.label2nodes[entity].copy()
        if current_:
            # Delete future events, you can see current events, before query time
            if self.dataset == 'YAGO':
                nodes = list(filter((lambda x: x[1] >= time), nodes))
            else: nodes = list(filter((lambda x: x[1] <= time), nodes))
        else:
            # No future events, no current events
            if self.dataset == 'YAGO':
                nodes = list(filter((lambda x: x[1] > time), nodes))
            else: nodes = list(filter((lambda x: x[1] < time), nodes))
        nodes.sort(key=lambda x: x[1], reverse=True)
        relation_space_high=[]
        dst_space_low_dict = {}
        for node in nodes:
            for src, dst, rel in self.graph.out_edges(node, data=True):

                if rel['relation'] in relation_space_high:
                    if max_action_num is None or len(dst_space_low_dict[rel['relation']]) < max_action_num:
                        dst_space_low_dict[rel['relation']].append((dst[0], dst[1]))
                elif max_option_num is None or len(relation_space_high) < max_option_num:
                    relation_space_high.append(rel['relation'])
                    dst_space_low_dict[rel['relation']] = [(dst[0], dst[1])]

                #
                #
                # relation_space_high.append(rel['relation'])
                # if rel['relation'] in dst_space_low_dict.keys():
                #     dst_space_low_dict[rel['relation']].append((dst[0], dst[1]))
                # else:
                #     dst_space_low_dict[rel['relation']] = [(dst[0], dst[1])]
        relation_space_high = list(set(relation_space_high))
        for key, value in dst_space_low_dict.items():
            value = np.array(list(set(value)), dtype=np.dtype('int32'))
            dst_space_low_dict[key] = value

        return np.array(relation_space_high, dtype=np.dtype('int32')), dst_space_low_dict

    def get_state_actions_space_complete(self, entity, time, current_=False):
        """Get the action space of the current state.
        """
        # if max_action_num is None:
        max_action_num = self.config['max_action_num']
        max_option_num = self.config['max_option_num']
        if self.state_action_space:
            if (entity, time, current_) in self.state_action_space_key:
                return self.state_action_space[(entity, time, current_)]
        nodes = self.label2nodes[entity].copy()
        if current_:
            # Delete future events, you can see current events, before query time
            if self.dataset == 'YAGO':
                nodes = list(filter((lambda x: x[1] >= time), nodes))
            else: nodes = list(filter((lambda x: x[1] <= time), nodes))
        else:
            # No future events, no current events
            if self.dataset == 'YAGO':
                nodes = list(filter((lambda x: x[1] > time), nodes))
            else: nodes = list(filter((lambda x: x[1] < time), nodes))
        nodes.sort(key=lambda x: x[1], reverse=True)
        relation_space_high=[]
        dst_space_low_dict = {}
        for node in nodes:
            for src, dst, rel in self.graph.out_edges(node, data=True):

                if rel['relation'] in relation_space_high:
                    if max_action_num is None or len(dst_space_low_dict[rel['relation']]) < max_action_num:
                        dst_space_low_dict[rel['relation']].append((dst[0], dst[1]))
                elif max_option_num is None or len(relation_space_high) < max_option_num:
                    relation_space_high.append(rel['relation'])
                    dst_space_low_dict[rel['relation']] = [(dst[0], dst[1])]

                #
                #
                # relation_space_high.append(rel['relation'])
                # if rel['relation'] in dst_space_low_dict.keys():
                #     dst_space_low_dict[rel['relation']].append((dst[0], dst[1]))
                # else:
                #     dst_space_low_dict[rel['relation']] = [(dst[0], dst[1])]
        relation_space_high = list(set(relation_space_high))
        for key, value in dst_space_low_dict.items():
            value = np.array(list(set(value)), dtype=np.dtype('int32'))
            dst_space_low_dict[key] = value

        return np.array(relation_space_high, dtype=np.dtype('int32')), dst_space_low_dict

    def next_actions(self, entites, times, query_times, max_action_num=200, first_step=False, source_to_dst=True):
        """get the option space of relation level
        """
        temp = times
        times = query_times
        query_times = temp

        if self.config['cuda']:
            entites = entites.cpu()
            times = times.cpu()
            query_times = times.cpu()

        entites = entites.numpy()
        times = times.numpy()
        query_times = query_times.numpy()

        options_high = self.get_padd_actions(entites, times, query_times, max_action_num, first_step, source_to_dst)

        if self.config['cuda']:
            options_high = torch.tensor(options_high, dtype=torch.long, device='cuda')
        else:
            options_high = torch.tensor(options_high, dtype=torch.long)
        return options_high

    def next_entities_space(self, entites, times, query_times, sample_rel, max_action_num, first_step=False):
        '''get the action space of entity level'''

        if self.config['cuda']:
            entites = entites.cpu()
            times = times.cpu()
            query_times = times.cpu()
            sample_rel = sample_rel.cpu()

        entites = entites.numpy()
        times = times.numpy()
        query_times = query_times.numpy()
        sample_rel = sample_rel.numpy()

        actions_low = self.get_padd_entities(entites, times, query_times, sample_rel, max_action_num, first_step)

        if self.config['cuda']:
            actions_low = torch.tensor(actions_low, dtype=torch.long, device='cuda')
        else:
            actions_low = torch.tensor(actions_low, dtype=torch.long)
        return actions_low

    def get_padd_actions(self, entites, times, query_times, max_action_num=200, first_step=False, source_to_dst=True):
        """Construct the model input array.
        If the optional actions are greater than the maximum number of actions, then sample,
        otherwise all are selected, and the insufficient part is pad.
        """
        max_relation_num = 0
        relation_list = []
        self.entities_list = []
        for i in range(entites.shape[0]):

            if times[i] == query_times[i]:
                relation_array, entity_array = self.get_state_actions_space_complete(entites[i], times[i], False)   #根据输入的源点(es,t)获得动作空间(r,eo,t)
            else:                                                                                   #(relation, eneity, time)
                relation_array, entity_array = self.get_state_actions_space_complete(entites[i], times[i], True)


            relation_list.append(relation_array)
            self.entities_list.append(entity_array)

            if relation_array.shape[0]>max_relation_num:
                max_relation_num = relation_array.shape[0]


        options_high = np.ones((entites.shape[0], max_relation_num+1, 1), dtype=np.dtype('int32'))

        options_high[:, :, 0] *= self.rPAD

        for i in range(entites.shape[0]):
            # NO OPERATION

            options_high[i, 0, 0] = self.NO_OP

            if relation_list[i].shape[0] == 0:
                continue

            # Whether to keep the action NO_OPERATION
            start_idx = 1
            if first_step:
                start_idx = 0


            options_high[i, start_idx:relation_list[i].shape[0]+start_idx, 0] = np.array(relation_list[i])


        return options_high

    def get_padd_entities(self, entites, times, query_times, sample_rels, max_action_num=200, first_step=False):

        assert len(sample_rels.shape) == 2
        max_entity_num = 0
        entities_list = []
        entity_batch_size = sample_rels.shape[0]*sample_rels.shape[1]
        entites = np.repeat(entites, entity_batch_size//entites.shape[0])
        times = np.repeat(times, entity_batch_size//times.shape[0])
        for i in range(sample_rels.shape[0]):   # batch_size
            for rel in sample_rels[i]:
                if rel in self.entities_list[i]:
                    entites_array = self.entities_list[i][rel]
                else:
                    entites_array = np.array([], dtype=np.dtype('int32'))
                if entites_array.shape[0] > max_entity_num:
                    max_entity_num = entites_array.shape[0]
                entities_list.append(entites_array)

        actions_low = np.ones((entity_batch_size, max_entity_num + 1, 2), dtype=np.dtype('int32'))
        actions_low[:, :, 0] *= self.ePAD
        actions_low[:, :, 1] *= self.tPAD

        for i in range(entity_batch_size):
            # NO OPERATION
            actions_low[i, 0, 0] = entites[i]
            actions_low[i, 0, 1] = times[i]

            if entities_list[i].shape[0] == 0:
                continue

            # Whether to keep the action NO_OPERATION
            start_idx = 1
            if first_step:
                # The first step cannot stay in place
                start_idx = 0


            actions_low[i, start_idx:entities_list[i].shape[0]+start_idx, ] = entities_list[i]

        return actions_low

    def abst_embds_productor(self, num_t, dim_t):
        '''get the absolute time coding using the position coding function of Transformer'''
        abst_embedding = np.zeros((num_t, dim_t))
        abst = np.expand_dims(np.arange(0, num_t), axis=-1)
        div_term = np.expand_dims(np.exp(np.arange(0, dim_t, 2, dtype=np.float32) * (-math.log(10000.0) / dim_t)), axis=0)

        abst_embedding[:, 0::2] = np.sin(abst * div_term)
        abst_embedding[:, 1::2] = np.cos(abst * div_term)

        return abst_embedding
