import torch
import torch.nn as nn
import numpy as np
import math

class HistoryEncoder_HL(nn.Module):
    '''lstm of relation level'''
    def __init__(self, config):
        super(HistoryEncoder_HL, self).__init__()
        self.config = config
        self.lstm_cell = torch.nn.LSTMCell(input_size=config['rel_dim'],
                                           hidden_size=config['state_dim'])

    def set_hiddenx(self, batch_size):
        """Set hidden layer parameters. Initialize to 0"""
        if self.config['cuda']:
            self.hx = torch.zeros(batch_size, self.config['state_dim'], device='cuda')
            self.cx = torch.zeros(batch_size, self.config['state_dim'], device='cuda')
        else:
            self.hx = torch.zeros(batch_size, self.config['state_dim'])
            self.cx = torch.zeros(batch_size, self.config['state_dim'])

    def forward(self, prev_action, mask):
        """mask: True if NO_OP. ON_OP does not affect history coding results"""
        self.hx_, self.cx_ = self.lstm_cell(prev_action, (self.hx, self.cx))
        self.hx = torch.where(mask, self.hx, self.hx_)
        self.cx = torch.where(mask, self.cx, self.cx_)
        return self.hx

class HistoryEncoder_LL(nn.Module):
    '''lstm of entity level'''
    def __init__(self, config):
        super(HistoryEncoder_LL, self).__init__()
        self.config = config
        self.lstm_cell = torch.nn.LSTMCell(input_size=config['ent_dim'],
                                           hidden_size=config['state_dim'])

    def set_hiddenx(self, batch_size):
        """Set hidden layer parameters. Initialize to 0"""
        if self.config['cuda']:
            self.hx = torch.zeros(batch_size, self.config['state_dim'], device='cuda')
            self.cx = torch.zeros(batch_size, self.config['state_dim'], device='cuda')
        else:
            self.hx = torch.zeros(batch_size, self.config['state_dim'])
            self.cx = torch.zeros(batch_size, self.config['state_dim'])

    def forward(self, prev_action, mask):
        """mask: True if NO_OP. ON_OP does not affect history coding results"""
        self.hx_, self.cx_ = self.lstm_cell(prev_action, (self.hx, self.cx))
        self.hx = torch.where(mask, self.hx, self.hx_)
        self.cx = torch.where(mask, self.cx, self.cx_)
        return self.hx

class PolicyMLP_High(nn.Module):
    def __init__(self, config):
        super(PolicyMLP_High, self).__init__()
        self.mlp_l1 = nn.Linear(config['mlp_input_dim_high'],
                                config['mlp_hidden_dim'], bias=True)
        self.mlp_l2 = nn.Linear(config['mlp_hidden_dim'], config['rel_dim'], bias=True)
        self.dropout0 = nn.Dropout(config['dp0_hl'])
        self.dropout1 = nn.Dropout(config['dp1_hl'])
        self.dropout2 = nn.Dropout(config['dp2_hl'])

    def forward(self, state_query):
        state_query = self.dropout0(state_query)
        hidden = torch.relu(self.mlp_l1(state_query))
        hidden = self.dropout1(hidden)
        output = self.mlp_l2(hidden).unsqueeze(1)
        output = self.dropout2(output)
        return output

class PolicyMLP_Low(nn.Module):
    def __init__(self, config):
        super(PolicyMLP_Low, self).__init__()
        self.mlp_l1 = nn.Linear(config['mlp_input_dim_low'],
                                config['mlp_hidden_dim'], bias=True)
        self.mlp_l2 = nn.Linear(config['mlp_hidden_dim'], config['ent_dim'], bias=True)
        self.dropout0 = nn.Dropout(config['dp0_ll'])
        self.dropout1 = nn.Dropout(config['dp1_ll'])
        self.dropout2 = nn.Dropout(config['dp2_ll'])

    def forward(self, agent_satet):

        agent_satet = self.dropout0(agent_satet)
        hidden = torch.relu(self.mlp_l1(agent_satet))
        hidden = self.dropout1(hidden)
        output = self.mlp_l2(hidden).unsqueeze(1)
        output = self.dropout2(output)
        return output

class DynamicEmbedding(nn.Module):
    def __init__(self, n_ent, dim_ent, dim_t, dataset, abst_embs):
        super(DynamicEmbedding, self).__init__()
        self.ent_embs = nn.Embedding(n_ent, dim_ent - dim_t)
        self.w = torch.nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, dim_t))).float())
        self.b = torch.nn.Parameter(torch.zeros(dim_t).float())

        '''absolute time'''
        self.dataset = dataset
        self.abst_embs = abst_embs
        self.t_w = nn.Parameter(-torch.ones(dim_t).float())

    def forward(self, entities, dt, abst):
        dt = dt.unsqueeze(-1)
        batch_size = dt.size(0)
        seq_len = dt.size(1)

        dt = dt.view(batch_size, seq_len, 1)

        repeat_t_w = torch.sigmoid(self.t_w).unsqueeze(0).unsqueeze(0)

        t = torch.cos(self.w.view(1, 1, -1) * dt + self.b.view(1, 1, -1))
        if len(entities.shape) == 1:
            t = t.squeeze(1)  # [batch_size, time_dim]
            repeat_t_w = repeat_t_w.squeeze(1)

        if self.dataset == 'YAGO':
            t = 0
        '''absolute time'''
        abst = abst.reshape(-1)
        if self.dataset in ("ICEWS14", "ICEWS18"):
            '''ICEWS14, ICEWS18'''
            abst = torch.div(abst, 24, rounding_mode='floor')
            # abst = abst//24
            abst_embd = self.abst_embs[abst]
        else:
            abst_embd = self.abst_embs[abst]
        abst_embd = abst_embd.reshape([batch_size, seq_len, -1])
        if len(entities.shape) == 1:
            abst_embd = abst_embd.squeeze(1)

        t = (1-repeat_t_w)*t+repeat_t_w*abst_embd

        e = self.ent_embs(entities)
        return torch.cat((e, t), -1)

class StaticEmbedding(nn.Module):
    def __init__(self, n_ent, dim_ent):
        super(StaticEmbedding, self).__init__()
        self.ent_embs = nn.Embedding(n_ent, dim_ent)

    def forward(self, entities):
        return self.ent_embs(entities)


class High_level_Agent(nn.Module):
    def __init__(self, config, rel_text_embd=None):
        super(High_level_Agent, self).__init__()
        self.num_rel = config['num_rel'] * 2 + 2
        self.config = config

        self.NO_OP = self.num_rel  # Stay in place; No Operation
        self.rPAD = config['num_rel'] * 2 + 1  # Padding relation
        self.real_rel_num = config['num_rel']

        self.pad_mask_hl = 0
        self.neighbors_relations = 0
        self.transit_high_agent_state = None
        self.relation_score = 0
        self.a = 0

        self.rel_embs = nn.Embedding(config['num_ent'], config['rel_dim'])

        self.policy_step = HistoryEncoder_HL(config)
        self.policy_mlp_high = PolicyMLP_High(config)


        self.rel_text_embd = None
        self.rel_text_emb_dim=0
        self.dataset = config['dataset']
        if rel_text_embd is not None:
            self.rel_text_embd = rel_text_embd
            self.rel_text_emb_dim = rel_text_embd.shape[-1]

    def forward(self, prev_relations, query_relation_embds, hl_space,
                query_rel=None):

        # embedding
        prev_rel_embds = self.rel_embs(prev_relations)
        pad_mask_hl = torch.ones_like(hl_space[:, :, 0]) * self.rPAD  # [batch_size, action_number]
        pad_mask_hl = torch.eq(hl_space[:, :, 0], pad_mask_hl)  # [batch_size, action_number]
        self.pad_mask_hl = pad_mask_hl

        # History Encode
        NO_OP_mask = torch.eq(prev_relations, torch.ones_like(prev_relations) * self.NO_OP)  # [batch_size]
        NO_OP_mask = NO_OP_mask.repeat(self.config['state_dim'], 1).transpose(1, 0)  # [batch_size, state_dim]
        lstm_output = self.policy_step(prev_rel_embds, NO_OP_mask)

        '''neigbors embedding'''
        neighbors_relations = self.rel_embs(hl_space[:, :, 0])  # [batch_size, action_num, rel_dim]
        self.neighbors_relations = neighbors_relations

        '''state'''
        self.transit_high_agent_state = torch.cat([lstm_output, query_relation_embds], dim=-1)

        '''score'''
        batch_size = neighbors_relations.shape[0]
        rel_size = neighbors_relations.shape[1]


        self.query_rel_text_emb = None
        if self.rel_text_embd is not None and query_rel is not None:
            self.query_rel_text_emb = self.rel_text_embd[query_rel].unsqueeze(1)
            neigh_rel_text_emb = self.rel_text_embd[hl_space.reshape(-1)].reshape([batch_size, rel_size,
                                                                                   self.rel_text_emb_dim])

            self.neighbors_relations = torch.cat([self.neighbors_relations, neigh_rel_text_emb], dim=-1)


    def transit_high(self, hl_space):
        '''relation level policy network and relation scorer'''
        chosen_relation = self.policy_mlp_high(self.transit_high_agent_state)

        logits_hl = self.get_score_hl(chosen_relation)
        sample_rel, sample_rel_idx = self.sample_high(hl_space)
        loss_hl = self.hl_loss(sample_rel_idx, logits_hl)
        return loss_hl, logits_hl, sample_rel

    def hl_loss(self, sample_rel_idx, logits_hl):
        one_hot_hl = torch.zeros_like(logits_hl).scatter(1, sample_rel_idx, 1)
        loss_hl = - torch.sum(torch.mul(logits_hl, one_hot_hl), dim=1)
        return loss_hl

    def get_score_hl(self, r_space):

        if self.query_rel_text_emb is not None:
            r_space = torch.cat([r_space, self.query_rel_text_emb], dim=-1)
        relation_score = torch.sum(torch.mul(self.neighbors_relations, r_space), dim=2)
        relation_score = (relation_score).masked_fill(self.pad_mask_hl, -1e10)
        self.relation_score = relation_score
        logits_hl = torch.nn.functional.log_softmax(relation_score, dim=1)
        return logits_hl

    def sample_high(self, hl_space):
        policy_disttr_relation = torch.softmax(self.relation_score, dim=1)
        next_r_idx = torch.multinomial(policy_disttr_relation, 1, replacement=True)  #[batch_size, 1]
        next_r = torch.gather(hl_space[:, :, 0], dim=1, index=next_r_idx).reshape(hl_space.shape[0])
        return next_r, next_r_idx

class Low_level_Agent(nn.Module):
    def __init__(self, config, abst_embs, ent_text_embd=None):
        super(Low_level_Agent, self).__init__()
        self.config = config

        self.ePAD = config['num_ent']  # Padding entity
        self.tPAD = 0  # Padding time
        self.NO_OP = config['num_rel'] * 2 + 2  # Stay in place; No Operation

        self.pad_mask_ll = 0
        self.neighbors_entities = 0
        self.transit_low_agent_state = None
        self.query_dst_score = 0.

        self.entities_score = 0
        self.b = 0

        if self.config['entities_embeds_method'] == 'dynamic':
            self.ent_embs = DynamicEmbedding(config['num_ent']+1, config['ent_dim'], config['time_dim'], config['dataset'], abst_embs)
        else:
            self.ent_embs = StaticEmbedding(config['num_ent']+1, config['ent_dim'])

        self.policy_step = HistoryEncoder_LL(config)
        self.policy_mlp_low = PolicyMLP_Low(config)

        self.score_weighted_fc_ll = nn.Linear(
            self.config['ent_dim'] * 2 + self.config['state_dim'],
            1, bias=True)

        self.ent_text_embd = None
        self.ent_text_emb_dim = 0
        if ent_text_embd is not None:
            self.ent_text_embd = ent_text_embd
            self.ent_text_emb_dim = ent_text_embd.shape[-1]


        self.time_decay = 0.0
        self.dataset=config['dataset']

    def forward(self, current_entities, current_timestamps,prev_relations,
                query_entity_embds, query_timestamps, sample_rel, ll_space,
                query_dst=None, query_text_node=None):

        # embedding
        current_delta_times = query_timestamps - current_timestamps
        current_ent_embds = self.ent_embs(current_entities, current_delta_times, current_timestamps)
        '''mask'''
        pad_mask_ll = torch.ones_like(ll_space[:, :, 0]) * self.ePAD
        pad_mask_ll = torch.eq(ll_space[:, :, 0], pad_mask_ll)
        self.pad_mask_ll = pad_mask_ll

        '''neighbor embedding'''
        dst_num = ll_space.size(1)
        rel_num = ll_space.size(0)
        if query_timestamps.shape[0] != rel_num:
            query_timestamps = query_timestamps.repeat(rel_num//query_timestamps.shape[0], 1).transpose(1,0).reshape(-1)
        neighbors_delta_time = query_timestamps.repeat(dst_num, 1).transpose(1,0) - ll_space[:, :, 1]
        neighbors_entities = self.ent_embs(ll_space[:, :, 0], neighbors_delta_time, ll_space[:, :, 1])  # [batch_size, action_num, ent_dim]
        self.neighbors_entities = neighbors_entities

        self.time_decay = self.get_time_decay(neighbors_delta_time, self.dataset, ll_space[:,:,1])

        if query_dst is not None:
            query_dst_emb = self.ent_embs(query_dst, torch.zeros_like(query_timestamps), query_timestamps)
            if query_dst_emb.shape[0] != rel_num:
                query_dst_emb = query_dst_emb.repeat(1, rel_num // query_dst_emb.shape[0]).reshape([rel_num, -1])
            query_dst_emb = query_dst_emb.repeat(1,1,dst_num).reshape([rel_num, dst_num, -1])
            query_score = torch.sum(torch.mul(self.neighbors_entities, query_dst_emb), dim=-1)
            self.query_dst_score = torch.softmax(query_score, dim=-1)

        '''History Encode'''
        NO_OP_mask = torch.eq(prev_relations, torch.ones_like(prev_relations) * self.NO_OP)  # [batch_size]
        NO_OP_mask = NO_OP_mask.repeat(self.config['state_dim'], 1).transpose(1, 0)  # [batch_size, state_dim]
        lstm_output = self.policy_step(current_ent_embds, NO_OP_mask)
        transit_low_agent_state = torch.cat([lstm_output, query_entity_embds], dim=-1)

        agent_state_repeat = transit_low_agent_state
        batch_size = agent_state_repeat.shape[0]
        if batch_size != rel_num:
            agent_state_repeat = agent_state_repeat.repeat(1, rel_num // batch_size).reshape(rel_num, -1)
        self.transit_low_agent_state = agent_state_repeat

        agent_state_repeat_ll = self.transit_low_agent_state.repeat(1, dst_num).reshape([rel_num, dst_num, -1])
        score_attention_input_ll = torch.cat([neighbors_entities, agent_state_repeat_ll], dim=-1)
        b = self.score_weighted_fc_ll(score_attention_input_ll)
        b = torch.sigmoid(b).squeeze(-1)
        self.b = b

        '''transformer'''
        self.query_dst_text_emb = None
        if self.ent_text_embd is not None and query_text_node is not None:
            self.query_dst_text_emb = self.ent_text_embd[query_text_node].unsqueeze(1)
            neigh_ent_text_emb = self.ent_text_embd[ll_space[:,:,0].reshape(-1)].reshape([rel_num, dst_num,
                                                                                          self.ent_text_emb_dim])

            self.neighbors_entities = torch.cat([self.neighbors_entities, neigh_ent_text_emb], dim=-1)

    def transit_low(self, ll_space):
        '''entity level policy network and entity scorer'''
        chosen_entity = self.policy_mlp_low(self.transit_low_agent_state)

        logits_ll = self.get_score_ll(chosen_entity,
                                      query_dst_score=self.query_dst_score)
        # logits_ll += e_local_loss
        sample_dst, sample_time, sample_dst_idx = self.sample_low(ll_space)
        # loss_ll = self.agent.ll_loss(sample_dst_idx, logits_ll) - e_local_loss
        loss_ll = self.ll_loss(sample_dst_idx, logits_ll)
        return loss_ll, logits_ll, sample_dst, sample_time

    def get_im_embedding(self, cooccurrence_entities):
        """Get the inductive mean representation of the co-occurrence relation.
        cooccurrence_entities: a list that contains the trained entities with the co-occurrence relation.
        return: torch.tensor, representation of the co-occurrence entities.
        """
        entities = self.ent_embs.ent_embs.weight.data[cooccurrence_entities]
        im = torch.mean(entities, dim=0)
        return im

    def update_entity_embedding(self, entity, ims, mu):
        """Update the entity representation with the co-occurrence relations in the last timestamp.
        entity: int, the entity that needs to be updated.
        ims: torch.tensor, [number of co-occurrence, -1], the IM representations of the co-occurrence relations
        mu: update ratio, the hyperparam.
        """
        self.source_entity = self.ent_embs.ent_embs.weight.data[entity]
        self.ent_embs.ent_embs.weight.data[entity] = mu * self.source_entity + (1 - mu) * torch.mean(ims, dim=0)

    def entities_embedding_shift(self, entity, im, mu):
        """Prediction shift."""
        self.source_entity = self.ent_embs.ent_embs.weight.data[entity]
        self.ent_embs.ent_embs.weight.data[entity] = mu * self.source_entity + (1 - mu) * im

    def back_entities_embedding(self, entity):
        """Go back after shift ends."""
        self.ent_embs.ent_embs.weight.data[entity] = self.source_entity

    def ll_loss(self, sample_dst_idx, logits_ll):
        one_hot_ll = torch.zeros_like(logits_ll).scatter(1, sample_dst_idx, 1)
        loss_ll = - torch.sum(torch.mul(logits_ll, one_hot_ll), dim=1)
        return loss_ll

    def get_score_ll(self, te_space, query_dst_score=0.0):

        if self.query_dst_text_emb is not None:
            te_space = torch.cat([te_space, self.query_dst_text_emb], dim=-1)
        entities_score = torch.sum(torch.mul(self.neighbors_entities, te_space), dim=2)  # [batch_size, action_number]
        entities_score = entities_score + query_dst_score
        entities_score -= self.time_decay
        entities_score = (self.b*entities_score).masked_fill(self.pad_mask_ll, -1e10)
        self.entities_score = entities_score
        logits_ll = torch.nn.functional.log_softmax(entities_score, dim=1)
        return logits_ll

    def sample_low(self, ll_space):
        policy_disttr_entity = torch.softmax(self.entities_score, dim=1)
        next_te_idx = torch.multinomial(policy_disttr_entity, 1, replacement=True)
        next_te = torch.gather(ll_space[:, :, 0], dim=1, index=next_te_idx).reshape(ll_space.shape[0])
        next_time = torch.gather(ll_space[:, :, 1], dim=1, index=next_te_idx).reshape(ll_space.shape[0])

        return next_te, next_time, next_te_idx

    def get_time_decay(self, dt, dataset, space_times):
        time_decay_rate = 0.001
        if dataset == 'ICEWS14':
            time_decay = torch.div(dt, 24, rounding_mode='floor')
            time_decay = time_decay_rate*time_decay
        elif dataset == 'ICEWS05-15':
            time_decay = time_decay_rate*dt
        elif dataset == 'YAGO':
            time_decay = -torch.softmax(space_times.float(), dim=-1)
        else:
            time_decay=0.0
        return time_decay

