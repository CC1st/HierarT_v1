import torch
import torch.nn as nn

class Episode(nn.Module):
    def __init__(self, env, agent_hl, agent_ll, config, kmeans_cluster=None):
        super(Episode, self).__init__()
        self.config = config
        self.env = env
        self.agent_hl = agent_hl
        self.agent_ll = agent_ll
        self.path_length = config['path_length']
        self.num_rel = config['num_rel']
        self.max_action_num = config['max_action_num']
        self.dataset = config['dataset']
        self.num_t = config['num_t']

        self.kmeans_cluster = kmeans_cluster

    def forward(self, query_entities, query_timestamps, query_relations, query_dst):
        """
        Args:
            query_entities: [batch_size]
            query_timestamps: [batch_size]
            query_relations: [batch_size]
        Return:
            all_loss: list
            all_logits: list
            all_actions_idx: list
            current_entities: torch.tensor, [batch_size]
            current_timestamps: torch.tensor, [batch_size]
        """

        # query_dst_emb = self.agent_ll.ent_embs(query_dst, torch.zeros_like(query_timestamps))

        # self.agent.entity2cluster.product_entity2cluster()
        # query_dst_labels = self.agent.entity2cluster.get_ent2cluster_for(query_dst)
        # query_rel_labels = self.agent.entity2cluster.get_rel2cluster(query_relations)


        query_entities_embeds = self.agent_ll.ent_embs(query_entities, torch.zeros_like(query_timestamps), query_timestamps)
        query_relations_embeds = self.agent_hl.rel_embs(query_relations)

        current_entites = query_entities
        current_timestamps = query_timestamps
        prev_relations = torch.ones_like(query_relations) * self.num_rel  # NO_OP

        all_loss_hl = []
        all_loss_ll = []
        all_logits_hl = []
        all_logits_ll = []
        cluster_reward = 0.


        self.agent_hl.policy_step.set_hiddenx(query_relations.shape[0])
        self.agent_ll.policy_step.set_hiddenx(query_entities.shape[0])
        for t in range(self.path_length):
            if t == 0:
                first_step = True
            else:
                first_step = False

            hl_space = self.env.next_actions(
                current_entites,
                current_timestamps,
                query_timestamps,
                self.max_action_num,
                first_step
            )


            self.agent_hl(prev_relations,
                          query_relations_embeds,
                          hl_space,
                          query_relations)

            loss_hl, logits_hl, sample_rel = self.agent_hl.transit_high(hl_space)

            ll_space = self.env.next_entities_space(current_entites,
                                                      current_timestamps,
                                                      query_timestamps,
                                                      sample_rel.unsqueeze(-1),
                                                      self.max_action_num,
                                                      first_step)


            self.agent_ll(current_entites, current_timestamps,prev_relations,
                          query_entities_embeds, query_timestamps,
                          sample_rel, ll_space,
                          query_dst=query_dst,
                          query_text_node=query_dst)

            loss_ll, logits_ll, sample_dst, sample_time = self.agent_ll.transit_low(ll_space)

            all_loss_hl.append(loss_hl)
            all_loss_ll.append(loss_ll)
            all_logits_hl.append(logits_hl)
            all_logits_ll.append(logits_ll)

            current_entites = sample_dst
            current_timestamps = sample_time
            prev_relations = sample_rel

        '''cluster labels'''
        if self.kmeans_cluster is not None:
            query_dst_labels = self.kmeans_cluster.get_ent2cluster(query_dst)
            query_timestamps_labels = self.kmeans_cluster.get_time_cluster(query_timestamps)

            current_entites_labels = self.kmeans_cluster.get_ent2cluster(current_entites)
            current_timestamps_labels = self.kmeans_cluster.get_time_cluster(current_timestamps)

            cluster_reward_dst = (query_dst_labels == current_entites_labels)
            cluster_reward_timestamps = (query_timestamps_labels == current_timestamps_labels)


            cluster_reward = (cluster_reward_dst * cluster_reward_timestamps).float() * 0.1


        res_dict={'all_loss_hl': all_loss_hl,
                  'all_loss_ll': all_loss_ll,
                  'all_logits_hl': all_logits_hl,
                  'all_logits_ll': all_logits_ll,
                  'current_entities': current_entites,
                  'current_timestamps': current_timestamps,
                  'all_cluster_reward': cluster_reward,
                  }

        return res_dict

    def beam_search(self, query_entities, query_timestamps, query_relations):
        '''beam search for test model'''

        batch_size = query_entities.shape[0]
        query_entities_embeds = self.agent_ll.ent_embs(query_entities, torch.zeros_like(query_timestamps), query_timestamps)
        query_relations_embeds = self.agent_hl.rel_embs(query_relations)

        hrl_logits_a = 0.6

        self.agent_hl.policy_step.set_hiddenx(batch_size)
        self.agent_ll.policy_step.set_hiddenx(batch_size)

        current_entites = query_entities
        current_timestamps = query_timestamps
        prev_relations = torch.ones_like(query_relations) * self.num_rel  # NO_OP

        hl_space = self.env.next_actions(current_entites,
                                                   current_timestamps,
                                                   query_timestamps,
                                                   self.max_action_num,
                                                   first_step=True)



        self.agent_hl(prev_relations, query_relations_embeds, hl_space)


        chosen_relation = self.agent_hl.policy_mlp_high(self.agent_hl.transit_high_agent_state)
        logits_hl = self.agent_hl.get_score_hl(chosen_relation)

        #relation level
        hl_space_size = hl_space.shape[1]
        if self.config['hl_beam_size'] > hl_space_size:
            hl_beam_size = hl_space_size
        else:
            hl_beam_size = self.config['hl_beam_size']
        hl_beam_log_prob, hl_top_k_id = torch.topk(logits_hl, hl_beam_size, dim=1)  # beam_log_prob.shape [batch_size, hl_beam_size]
        hl_beam_log_prob = hl_beam_log_prob.reshape(-1)  # [batch_size * beam_size]

        sample_rel = torch.gather(hl_space[:, :, 0], dim=1, index=hl_top_k_id)
        hl_space_beam = sample_rel.reshape(-1)

        #entity level
        ll_space = self.env.next_entities_space(current_entites,
                                                  current_timestamps,
                                                  query_timestamps,
                                                  sample_rel,
                                                  self.max_action_num,
                                                  first_step=True
                                                  )

        self.agent_ll(current_entites, current_timestamps, prev_relations,
                      query_entities_embeds, query_timestamps, hl_space_beam, ll_space)


        chosen_entity = self.agent_ll.policy_mlp_low(self.agent_ll.transit_low_agent_state) #[batch_size*hl_beam_size, actions_num, entity_dim]
        logits_ll = self.agent_ll.get_score_ll(chosen_entity)
        ll_space_size = ll_space.shape[1]
        if self.config['ll_beam_size'] > ll_space_size:
            ll_beam_size = ll_space_size
        else:
            ll_beam_size = self.config['ll_beam_size']
        ll_beam_log_prob, ll_top_k_id = torch.topk(logits_ll, ll_beam_size, dim=1)  #[batch_size*hl_beam_size, ll_beam_size]
        ll_beam_log_prob = ll_beam_log_prob.reshape([batch_size, -1])  #[batch_size, hl_beam_size*ll_beam_size]
        ll_space_beam_entity = torch.gather(ll_space[:, :, 0], dim = 1, index=ll_top_k_id)
        ll_space_beam_time = torch.gather(ll_space[:, :, 1], dim=1, index=ll_top_k_id)


        hl_beam_log_prob = hl_beam_log_prob.repeat(ll_beam_size, 1)
        hl_beam_log_prob = hl_beam_log_prob.transpose(1, 0).reshape([batch_size, -1])
        logits = (1-hrl_logits_a)*ll_beam_log_prob + hrl_logits_a*hl_beam_log_prob

        #beam search the low level results
        beam_size = logits.shape[1]
        if self.config['beam_size'] < beam_size:
            beam_size = self.config['beam_size']
        beam_log_prob, top_k_id = torch.topk(logits, beam_size, dim=1) #[batch_size, beam_size]
        # beam_log_prob = beam_log_prob.reshape(-1)   #[batch_size*beam_size]


        hl_beam_log_prob = torch.gather(hl_beam_log_prob, dim=1, index=top_k_id).reshape(-1)
        ll_beam_log_prob = torch.gather(ll_beam_log_prob, dim=1, index=top_k_id).reshape(-1)



        current_entites = torch.gather(ll_space_beam_entity.reshape([batch_size, -1]),
                                      dim=1, index=top_k_id).reshape(-1)
        current_timestamps = torch.gather(ll_space_beam_time.reshape([batch_size, -1]),
                                          dim=1, index=top_k_id).reshape(-1)  # [batch_size * beam_size]
        prev_relations = torch.gather(hl_space_beam.repeat(ll_beam_size, 1).transpose(1,0).reshape([batch_size, -1]),
                                       dim=1, index=top_k_id).reshape(-1)  # [batch_size * beam_size]


        self.agent_hl.policy_step.hx = self.agent_hl.policy_step.hx.repeat(1, 1, beam_size).reshape([batch_size*beam_size, -1])
        self.agent_hl.policy_step.cx = self.agent_hl.policy_step.cx.repeat(1, 1, beam_size).reshape([batch_size*beam_size, -1])
        self.agent_ll.policy_step.hx = self.agent_ll.policy_step.hx.repeat(1, 1, beam_size).reshape([batch_size*beam_size, -1])
        self.agent_ll.policy_step.cx = self.agent_ll.policy_step.cx.repeat(1, 1, beam_size).reshape([batch_size*beam_size, -1])


        for t in range(1, self.path_length):
            query_timestamps_roll = query_timestamps.repeat(beam_size, 1).permute(1, 0).reshape(-1)
            query_entities_embeds_roll = query_entities_embeds.repeat(1, 1, beam_size)
            query_entities_embeds_roll = query_entities_embeds_roll.reshape(
                [batch_size * beam_size, -1])  # [batch_size * beam_size, ent_dim]
            query_relations_embeds_roll = query_relations_embeds.repeat(1, 1, beam_size)
            query_relations_embeds_roll = query_relations_embeds_roll.reshape(
                [batch_size * beam_size, -1])  # [batch_size * beam_size, rel_dim]


            hl_space = self.env.next_actions(current_entites,
                                                current_timestamps,
                                                query_timestamps_roll,
                                                self.max_action_num)  #[batch_size, actions_num, ]


            self.agent_hl(prev_relations, query_relations_embeds_roll, hl_space)

            # transit_high_level
            chosen_relation = self.agent_hl.policy_mlp_high(self.agent_hl.transit_high_agent_state) #[batch_size, action_num, rel_dim] [30, 100]

            logits_hl = self.agent_hl.get_score_hl(chosen_relation) #[batch_size, action_num]

            #relation level
            assert len(hl_beam_log_prob.shape) == 1
            hl_space_size = hl_space.shape[1]
            beam_tmp_hl = hl_beam_log_prob.repeat(hl_space_size, 1).transpose(1, 0) # [batch_size * hl_beam_size, max_action_num]
            try:
                beam_tmp_hl += logits_hl
            except:
                print(beam_tmp_hl, logits_hl)

            if beam_tmp_hl.shape[1] > self.config['hl_beam_size']:
                hl_beam_size = self.config['hl_beam_size']
            else:
                hl_beam_size = beam_tmp_hl.shape[1]


            top_k_log_prob_hl, top_k_action_id_hl = torch.topk(beam_tmp_hl, hl_beam_size, dim=1)  # [batch_size*hl_beam, hl_beam_size]

            sample_rel = torch.gather(hl_space[:, :, 0], dim=1, index=top_k_action_id_hl)
            hl_space_beam = sample_rel.reshape(-1)

            ll_space = self.env.next_entities_space(current_entites,
                                                       current_timestamps,
                                                       query_timestamps_roll,
                                                       sample_rel,
                                                       self.max_action_num)

            self.agent_ll(current_entites,
                          current_timestamps,
                          prev_relations,
                          query_entities_embeds_roll,
                          query_timestamps_roll,
                          hl_space_beam,
                          ll_space)

            ll_space_size = ll_space.shape[1]
            chosen_entity = self.agent_ll.policy_mlp_low(self.agent_ll.transit_low_agent_state) #[batch_size*hl_beam_size^2, actions_num, entity_dim]
            logits_ll = self.agent_ll.get_score_ll(chosen_entity)  # [batch_size*hl_beam_size^2, action_num]

            hx_tmp_hl = self.agent_hl.policy_step.hx.reshape(batch_size, beam_size, -1)
            cx_tmp_hl = self.agent_hl.policy_step.cx.reshape(batch_size, beam_size, -1)
            hx_tmp_ll = self.agent_ll.policy_step.hx.reshape(batch_size, beam_size, -1)
            cx_tmp_ll = self.agent_ll.policy_step.cx.reshape(batch_size, beam_size, -1)


            assert len(ll_beam_log_prob.shape) == 1
            beam_tmp_ll = ll_beam_log_prob.repeat(hl_beam_size, 1).transpose(1, 0).reshape(-1)  #[batch_size**beam_size*hl_beam_size]
            beam_tmp_ll = beam_tmp_ll.repeat(ll_space_size, 1).transpose(1, 0)   #[batch_size**beam_size*hl_beam_size, ll_space_size]

            beam_tmp_ll += logits_ll

            #beam_tmp = beam_tmp.reshape([1, -1])
            if beam_tmp_ll.shape[1] > self.config['ll_beam_size']:
                ll_beam_size = self.config['ll_beam_size']
            else:
                ll_beam_size = beam_tmp_ll.shape[1]
            # [beam_size*hl_beam_size, ll_beam_size]
            top_k_log_prob_ll, top_k_action_id_ll = torch.topk(beam_tmp_ll, ll_beam_size, dim=1)  # [batch_size*beam_size*hl_beam_size, ll_beam_size]

            ll_space_beam_entity = torch.gather(ll_space[:, :, 0], dim=1, index=top_k_action_id_ll)
            ll_space_beam_time = torch.gather(ll_space[:, :, 1], dim=1, index=top_k_action_id_ll)

            top_k_log_prob_hl = top_k_log_prob_hl.reshape(-1)
            top_k_log_prob_hl = top_k_log_prob_hl.repeat(ll_beam_size, 1).transpose(1,0) # [batch_size*beam_size*hl_beam_size, ll_beam_size]
            top_k_log_prob_hl = top_k_log_prob_hl.reshape(batch_size, -1)
            top_k_log_prob_ll = top_k_log_prob_ll.reshape(batch_size, -1)
            logits = (1-hrl_logits_a)*top_k_log_prob_ll + hrl_logits_a*top_k_log_prob_hl   # [batch_size*beam_size*hl_beam_size, ll_beam_size]

            # beam search k results of low level
            if logits.shape[1] >= self.config['beam_size']:
                beam_size = self.config['beam_size']
            else:
                beam_size = logits.shape[1]
            top_k_log_prob, top_k_action_id = torch.topk(logits, beam_size, dim=1)  # [batch_size, beam_size]
            offset = top_k_action_id // (hl_beam_size*ll_beam_size)
            offset = offset.unsqueeze(-1).repeat(1, 1, self.config['state_dim']) #[batch_size, beam_size]


            hl_beam_log_prob = torch.gather(top_k_log_prob_hl, dim=-1, index=top_k_action_id).reshape(-1)
            ll_beam_log_prob = torch.gather(top_k_log_prob_ll, dim=-1, index=top_k_action_id).reshape(-1)



            self.agent_hl.policy_step.hx = torch.gather(hx_tmp_hl, dim=1, index=offset)
            self.agent_hl.policy_step.hx = self.agent_hl.policy_step.hx.reshape([batch_size * beam_size, -1])
            self.agent_hl.policy_step.cx = torch.gather(cx_tmp_hl, dim=1, index=offset)
            self.agent_hl.policy_step.cx = self.agent_hl.policy_step.cx.reshape([batch_size * beam_size, -1])

            self.agent_ll.policy_step.hx = torch.gather(hx_tmp_ll, dim=1, index=offset)
            self.agent_ll.policy_step.hx = self.agent_ll.policy_step.hx.reshape([batch_size * beam_size, -1])
            self.agent_ll.policy_step.cx = torch.gather(cx_tmp_ll, dim=1, index=offset)
            self.agent_ll.policy_step.cx = self.agent_ll.policy_step.cx.reshape([batch_size * beam_size, -1])

            prev_relations = torch.gather(hl_space_beam.repeat(ll_beam_size, 1).transpose(1,0).reshape([batch_size, -1]),
                                          dim=1, index=top_k_action_id).reshape(-1)
            current_entites = torch.gather(ll_space_beam_entity.reshape([batch_size, -1]),
                                           dim=1, index=top_k_action_id).reshape(-1)
            current_timestamps = torch.gather(ll_space_beam_time.reshape([batch_size, -1]),
                                              dim=1, index=top_k_action_id).reshape(-1)


        return ll_space_beam_entity.reshape([batch_size, -1]), logits

