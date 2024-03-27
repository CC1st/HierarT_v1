import torch
import json
import os
import tqdm
import numpy as np

class Trainer(object):
    def __init__(self, model, pg, optimizer_hl, optimizer_ll, args, distribution=None):
        self.model = model#Episode(先验知识：动态entity+静态realtion，路径编码LSTM，MLP计算候选action score)
        self.pg = pg
        self.optimizer_hl = optimizer_hl
        self.optimizer_ll = optimizer_ll
        self.args = args
        self.distribution = distribution

    def train_epoch(self, dataloader, ntriple):
        self.model.train()
        total_loss = 0.0
        total_reward = 0.0
        counter = 0
        with tqdm.tqdm(total=ntriple, unit='ex') as bar:
            bar.set_description('Train')
            for src_batch, rel_batch, dst_batch, time_batch in dataloader:
                if self.args.cuda:
                    src_batch = src_batch.cuda()
                    rel_batch = rel_batch.cuda()
                    dst_batch = dst_batch.cuda()
                    time_batch = time_batch.cuda()

                #all_loss, all_logits, _, current_entities, current_time = self.model(src_batch, time_batch, rel_batch)
                #此处返回值中被忽略的两个参数是all_relation_idx, all_dstentity_idx
                res_dict = self.model(src_batch, time_batch, rel_batch, dst_batch)
                all_loss_hl = res_dict['all_loss_hl']
                all_loss_ll = res_dict['all_loss_ll']
                all_logits_hl = res_dict['all_logits_hl']
                all_logits_ll = res_dict['all_logits_ll']
                current_entities = res_dict['current_entities']
                current_time = res_dict['current_timestamps']
                all_cluster_reward = res_dict['all_cluster_reward']
                #验证当前预测entity是否为目标entity，如果是reward=1, 否则为0
                reward = self.pg.get_reward(current_entities, dst_batch)
                reward += all_cluster_reward
                reward[reward>1.0] = 1.0
                #reward_hl, reward_ll  = self.pg.get_reward(current_entities, dst_batch)
                if self.args.reward_shaping:
                    # reward shaping
                    delta_time = time_batch - current_time
                    p_dt = []

                    for i in range(rel_batch.shape[0]):
                        rel = rel_batch[i].item()
                        dt = delta_time[i].item()
                        p_dt.append(self.distribution(rel, dt // self.args.time_span))  #狄利克雷分布

                    p_dt = torch.tensor(p_dt)
                    if self.args.cuda:
                        p_dt = p_dt.cuda()
                    shaped_reward = (1 + p_dt) * reward #公式2, 计算query所得结果的reward
                    # shaped_reward += all_cluster_reward*0.1
                    #shaped_reward_ll = (1 + p_dt) * reward_ll #公式2, 计算query所得结果的reward
                    cum_discounted_reward = self.pg.calc_cum_discounted_reward(shaped_reward)
                else:
                    cum_discounted_reward = self.pg.calc_cum_discounted_reward(reward)
                # cum_discounted_cluster_reward = self.pg.calc_cum_discounted_reward(all_cluster_reward, gamma=0.95)
                # cum_discounted_reward += cum_discounted_cluster_reward*0.1
                # all_local_loss_repeat = torch.tensor(np.array(
                #     [loss.cpu().detach().numpy() for loss in all_local_loss])).cuda().t()
                # cum_discounted_reward += all_local_loss_repeat
                #cum_discounted_reward_hl = cum_discounted_reward_ll = cum_discounted_reward
                #reinfore_loss = self.pg.calc_reinforce_loss(all_loss, all_logits, cum_discounted_reward)
                reinfore_loss_hl = self.pg.calc_reinforce_loss(all_loss_hl, all_logits_hl, cum_discounted_reward)
                reinfore_loss_ll = self.pg.calc_reinforce_loss(all_loss_ll, all_logits_ll, cum_discounted_reward)
                #cum_discounted_reward = cum_discounted_reward_hl + cum_discounted_reward_ll
                reinfore_loss = reinfore_loss_hl + reinfore_loss_ll
                self.pg.baseline.update(torch.mean(cum_discounted_reward))
                self.pg.now_epoch += 1

                self.optimizer_hl.zero_grad()
                reinfore_loss_hl.backward()
                self.optimizer_ll.zero_grad()
                reinfore_loss_ll.backward()
                # self.optimizer.zero_grad()
                # reinfore_loss.backward()
                if self.args.clip_gradient:
                    # total_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_gradient)
                    total_norm_hl = torch.nn.utils.clip_grad_norm_(self.model.agent_hl.parameters(), self.args.clip_gradient)
                    total_norm_ll = torch.nn.utils.clip_grad_norm_(self.model.agent_ll.parameters(), self.args.clip_gradient)
                # self.optimizer.step()
                self.optimizer_hl.step()
                self.optimizer_ll.step()

                total_loss += reinfore_loss
                total_reward += torch.mean(reward)
                counter += 1
                bar.update(self.args.batch_size)
                bar.set_postfix(loss='%.4f' % reinfore_loss, reward='%.4f' % torch.mean(reward).item())
        return total_loss / counter, total_reward / counter

    def save_model(self, checkpoint_path='checkpoint.pth'):
        """Save the parameters of the model and the optimizer,"""
        argparse_dict = vars(self.args)
        with open(os.path.join(self.args.save_path, 'config.json'), 'w') as fjson:
            json.dump(argparse_dict, fjson)

        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_hl_state_dict': self.optimizer_hl.state_dict(),
            'optimizer_ll_state_dict': self.optimizer_ll.state_dict()},
            os.path.join(self.args.save_path, checkpoint_path)
        )
