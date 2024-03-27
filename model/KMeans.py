from sklearn.cluster import KMeans
import pickle
import os
import torch
import torch.nn as nn
import numpy as np


class Ent2Cluster(nn.Module):
    def __init__(self, config):
        super(Ent2Cluster, self).__init__()
        self.ent_k = config['train_entity_k']
        self.rel_k = config['train_rel_k']
        self.time_k = config['time_k']
        self.config = config
        self.real_rel_num = config['num_rel']
        self.ePAD = config['num_ent']  # Padding entity
        self.rPAD = config['num_rel'] * 2 + 1  # Padding relation
        self.tPAD = 0  # Padding time
        self.train_data_path = config['train_data_path']
        self.valid_data_path = config['valid_data_path']
        self.test_data_path = config['test_data_path']
        self.data_path = config['data_path']
        self.best_mrr = 0.
        # self.entity_all_idx = torch.tensor(np.arange(self.ePAD +1)).cuda()
        # self.relation_all_idx = torch.tensor(np.arange(self.rPAD +1)).cuda()
        # self.ent2cluster, self.rel2cluster = self.product_entity2cluster_dict(config['train_data_path'],
        #                                                                                   config['valid_data_path'],
        #                                                                                   config['test_data_path'],
        #                                                                                   config['train_entity_k'],
        #                                                                                   config['train_rel_k'])
        # self.valid_ent2cluster, self.valid_rel2cluster = self.product_entity2cluster_dict(config['valid_data_path'],
        #                                                                                   config['valid_entity_k'],
        #                                                                                   config['valid_rel_k'])
        # self.test_ent2cluster, self.test_rel2cluster = self.product_entity2cluster_dict(config['test_data_path'],
        #                                                                                 config['test_entity_k'],
        #                                                                                 config['test_rel_k'])
        # self.ent2cluster_key = torch.tensor(list(self.ent2cluster.keys())).cuda()
        # self.ent2cluster_value = torch.tensor(list(self.ent2cluster.values())).cuda()
        # self.rel2cluster_key = torch.tensor(list(self.rel2cluster.keys())).cuda()
        # self.rel2cluster_value = torch.tensor(list(self.rel2cluster.values())).cuda()
        # self.product_entity2cluster()
        # self.product_triple_file()
        # self.product_cluster_file_id()
        # if config['restore_cluster_file']:
        #     self.product_entity2cluster()   # product the files of entities or relations to clusters mapping
        # self.ent2cluster, self.rel2cluster, self.time2cluster = self.load_cluster()
        # self.ent2cluster_key = torch.tensor(list(self.ent2cluster.keys())).cuda()
        # self.ent2cluster_value = torch.tensor(list(self.ent2cluster.values())).cuda()
        # self.rel2cluster_key = torch.tensor(list(self.rel2cluster.keys())).cuda()
        # self.rel2cluster_value = torch.tensor(list(self.rel2cluster.values())).cuda()
        # self.time2cluster_key = torch.tensor(list(self.time2cluster.keys())).cuda()
        # self.time2cluster_value = torch.tensor(list(self.time2cluster.values())).cuda()

        # self.product_cluster_file_id()
        self.ent_cluster = None
        self.rel_cluster = None
        self.dataset = config['dataset']

        self.ent2cluster, self.time2cluster = self.product_entity2cluster()
        self.ent2cluster_key = torch.tensor(list(self.ent2cluster.keys())).cuda()
        self.ent2cluster_value = torch.tensor(list(self.ent2cluster.values())).cuda()
        self.time2cluster_key = torch.tensor(list(self.time2cluster.keys())).cuda()
        self.time2cluster_value = torch.tensor(list(self.time2cluster.values())).cuda()


    def get_ent2cluster(self, entities):

        entities_flatten = entities.view(-1)

        entity_labels = self.ent2cluster_key.unsqueeze(0) == entities_flatten.unsqueeze(1)

        ent2cluster_value_repeat = self.ent2cluster_value.repeat(entity_labels.shape[0], 1)
        entity_labels = ent2cluster_value_repeat[entity_labels]

        entity_labels = entity_labels.reshape(entities.shape)

        return entity_labels


    def product_cluster_file_id(self):

        train_entities = []
        relations = []
        triple_datas = []
        times = []
        with open(self.train_data_path, "r") as f:
            for line in f:
                line_split = line.split()
                head_time = (int(line_split[0]))
                tail_time = (int(line_split[2]))
                relations.append(int((line_split[1])))
                # relations.append(int((line_split[1])) + self.real_rel_num+1)
                train_entities.append(head_time)
                train_entities.append(tail_time)
                times.append(int(line_split[3]))

        with open(self.valid_data_path, "r") as f:
            for line in f:
                line_split = line.split()  # head rel tail time
                head_time = (int(line_split[0]))
                tail_time = (int(line_split[2]))
                relations.append(int((line_split[1])))
                train_entities.append(head_time)
                train_entities.append(tail_time)
                times.append(int(line_split[3]))

        with open(self.test_data_path, "r") as f:
            for line in f:
                line_split = line.split()  # head rel tail time
                head_time = (int(line_split[0]))
                tail_time = (int(line_split[2]))
                relations.append(int((line_split[1])))
                train_entities.append(head_time)
                train_entities.append(tail_time)
                times.append(int(line_split[3]))


        times.append(self.tPAD)
        times = list(set(times))
        delta_times = []
        for i in times:
            for j in times:
                delta_times.append( i -j)
        delta_times = list(set(delta_times))
        # timesFile = os.path.join(self.data_path, 'times.txt')
        # with open(timesFile, 'w') as f:
        #     for time in times:
        #         f.write(str(time) + '\n')
        train_entities.append(self.ePAD)
        train_entities = list(set(train_entities))

        train_entities = torch.tensor(train_entities).cuda().unsqueeze(-1)
        delta_times = torch.tensor(delta_times).cuda().unsqueeze(-1)
        entity_deltaTime2emb = {}
        for entity in train_entities:
            for time in delta_times:
                entity_deltaTime2emb[(entity.item(), time.item())] = self.ent_emb(entity, time)

        ent2dynamic_emb_file = os.path.join(self.data_path, 'ent2DynamicEmb.pkl')
        pickle.dump(entity_deltaTime2emb, open(ent2dynamic_emb_file, 'wb'))


        relations = np.array(relations)
        relations = np.append(relations, self.real_rel_num)
        relations_rev = relations + self.real_rel_num + 1
        relations = np.append(relations, relations_rev, axis=-1)
        # relations = np.append(relations, self.rPAD)
        relations = np.unique(relations)

        kmeans_entity = KMeans(n_clusters=self.ent_k, random_state=0).fit(train_entities)
        kmeans_relation = KMeans(n_clusters=self.rel_k, random_state=0).fit(np.expand_dims(relations, axis=-1))

        entity2cluster = dict(zip(train_entities, kmeans_entity.labels_))
        rel2cluster = dict(zip(relations, kmeans_relation.labels_))

        ent2clusterFile = os.path.join(self.data_path, 'entity2cluster.pkl')
        pickle.dump(entity2cluster, open(ent2clusterFile, 'wb'))

        rel2clusterFile = os.path.join(self.data_path, 'relation2cluster.pkl')
        pickle.dump(rel2cluster, open(rel2clusterFile, 'wb'))

    def product_entity2cluster(self):
        # def product_entity2cluster_dict(self, train_data_path, valid_data_path, test_data_path, ent_k, rel_k):
        # with open(data_path, "r") as f:
        #     train_entities = []
        #     for line in f:
        #         line_split = line.split()  # head rel tail time
        #         head_time = torch.tensor([int(line_split[0]), int(line_split[3])])
        #         tail_time = torch.tensor([int(line_split[2]), int(line_split[3])])
        #         train_entities.append(head_time)
        #         train_entities.append(tail_time)
        # train_entities_tensor = torch.stack(train_entities, dim=0)
        # train_entities = torch.unique(train_entities_tensor, dim=0, sorted=False)
        # train_entities = train_entities.cuda()
        #
        # train_entities_emb = self.ent_embs(train_entities[:,0], train_entities[:,1])
        #
        # # train_entity_list = np.array(train_entities.keys())
        # kmeans_entity = KMeans(n_clusters=self.k, random_state=0).fit(train_entities_emb.detach().cpu().numpy())
        #
        # entity2cluster = {}
        # train_entities = train_entities.detach().cpu().numpy()
        # train_entities_keys = zip(train_entities[:,0], train_entities[:,1])
        # for idx, ent_t in enumerate(train_entities_keys):
        #     entity2cluster[ent_t] = kmeans_entity.labels_[idx]  # entity index 与其所在聚类的lebal绑定
        #
        # # pickle.dump(entity2cluster, open(entity2cluster_path, 'wb'))
        # return entity2cluster
        entity2id = []

        # # /data/data-home/luoxuewei/titer_-time-traveler-master_hrl_DA/data/ICEWS14/
        # data_file_path = self.data_path
        data_file_path = self.data_path
        with open(os.path.join(data_file_path, 'Tucker/entity2id.txt'), "r") as f:
            for line in f:
                line_split = line.split()   # entity id
                entity2id.append([int(line_split[0]), int(line_split[1])])
        # with open(os.path.join(data_file_path, 'Tucker/relation2id.txt'), "r") as f:
        #     for line in f:
        #         line_split = line.split()   # relation id
        #         relation2id.append([int(line_split[0]), int(line_split[1])])
        # with open(os.path.join(data_file_path, 'times.txt'), "r") as f:
        #     for line in f:
        #         line_split = line.split()   # relation id
        #         times.append(int(line_split[0]))
        # times = np.array(times)

        # data_path = '/home/luoxuewei/Project/titer_-time-traveler-master_hrl_v3/data/ICEWS14/'
        entity_emb = torch.load(os.path.join(self.data_path, 'Tucker/ent2emb.pth'))
        # relation_emb = torch.load(os.path.join(self.data_path, 'Tucker/rel2emb.pth'))
        # abst_emb = torch.load(os.path.join(self.data_path, 'Tucker/abst2emb.pth'))
        abst_embds_path = os.path.join(self.data_path, 'Tucker/abst2emb.pkl')
        abst_emb = pickle.load(open(abst_embds_path, 'rb'))


        kmeans_entity = KMeans(n_clusters=self.ent_k, random_state=0).fit(entity_emb.detach().cpu().numpy())
        kmeans_abst = KMeans(n_clusters=self.time_k, random_state=0).fit(abst_emb)


        entity2cluster = {}
        time2cluster = {}
        for ent2id in entity2id:
            entity2cluster[ent2id[0]] = kmeans_entity.labels_[ent2id[1]]

        if self.dataset in ('ICEWS14', 'ICEWS18'):
            for idx, label in enumerate(kmeans_abst.labels_):
                time2cluster[idx*24] = label
        else:
            for idx, label in enumerate(kmeans_abst.labels_):
                time2cluster[idx] = label

        return entity2cluster, time2cluster

    def product_ent_cluster_by_selfEmb(self, entities, relative_times):

        ''' get dynamic entity embedding'''
        ent_embs = self.ent_emb(entities, relative_times)

        '''kmeans'''
        ent_cluster = KMeans(n_clusters=self.ent_k, random_state=0).fit(ent_embs.detach().cpu().numpy())

        ent_labels = torch.tensor(ent_cluster.labels_).cuda()

        return ent_labels

    def product_cluster_everyValid_by_selfEmb(self):

        # ent_emb_weights = self.ent_emb.ent_embs.weight.data
        ent_emb_weights = self.ent_emb.ent_embs(self.entity_all_idx)

        ent_cluster = KMeans(n_clusters=self.ent_k, random_state=0).fit(ent_emb_weights.detach().cpu().numpy())


        # rel_emb_weights = self.rel_emb.weight.data
        rel_emb_weights = self.rel_emb(self.relation_all_idx)

        rel_cluster = KMeans(n_clusters=self.rel_k, random_state=0).fit(rel_emb_weights.detach().cpu().numpy())

        # timestamp_emb
        t = self.delta_times
        t = t.view(t.shape[0], 1, 1)
        t = torch.cos(self.ent_emb.w.view(1, 1, -1) * t + self.ent_emb.b.view(1, 1, -1))
        t = t.squeeze(1)
        t = t.detach().cpu().numpy()

        kmeans_time = KMeans(n_clusters=self.time_k, random_state=0).fit(t)

        delta_time2cluster = {}
        for idx, label in enumerate(kmeans_time.labels_):
            delta_time2cluster[self.delta_times[idx].item()] = label


        ent_cluster_labels = ent_cluster.labels_
        rel_cluster_labels = rel_cluster.labels_
        if self.config['cuda']:
            ent_cluster_labels = torch.tensor(ent_cluster.labels_).cuda()
            rel_cluster_labels = torch.tensor(rel_cluster.labels_).cuda()

        self.ent_cluster = ent_cluster_labels
        self.rel_cluster = rel_cluster_labels

        return delta_time2cluster

        # return ent_cluster_labels, rel_cluster_labels

    def get_cluster_by_idx(self, index, is_entity=True):
        '''
        return cluster labels of inputting index
        :param index:
        :param is_entity:
        :return:
        '''

        index = index.view(-1)
        if self.config['cuda']:
            index = index.cpu()
        index = index.numpy()

        if is_entity:
            labels = self.ent_cluster[index]

        if not is_entity:
            '''relation'''
            labels = self.rel_cluster[index]

        return labels

    def get_time_cluster(self, times):

        times_flatten = times.view(-1)
        time_labels = self.time2cluster_key.unsqueeze(0) == times_flatten.unsqueeze(1)
        time2cluster_value_repeat = self.time2cluster_value.repeat(time_labels.shape[0], 1)
        time_labels =  time2cluster_value_repeat[time_labels]
        time_labels = time_labels.reshape(times.shape)
        return time_labels


    def load_time_cluster(self):

        '''ICEWS14'''
        times = []
        with open(os.path.join(self.data_path, 'times.txt'), "r") as f:
            for line in f:
                line_split = line.split()  # relation id
                times.append(int(line_split[0]))
        times = np.array(times)

        kmeans_time = KMeans(n_clusters=self.time_k, random_state=0).fit(np.expand_dims(times, axis=-1))

        time2cluster = {}
        for idx, label in enumerate(kmeans_time.labels_):
            time2cluster[times[idx]] = label

        # time2clusterFile = os.path.join(self.data_path, 'time2cluster.pkl')
        # time2cluster = pickle.load(open(time2clusterFile, 'rb'))
        return time2cluster

    def load_delta_time(self):

        delta_times = []
        with open(os.path.join(self.data_path, 'delta_time.txt'), 'r') as f:
            for line in f:
                delta_time = line.split()
                delta_times.append(int(delta_time[0]))

        return delta_times

    def product_deltaTime_file(self):

        times = []
        with open(os.path.join(self.data_path, 'times.txt'), "r") as f:
            for line in f:
                line_split = line.split()  # relation id
                times.append(int(line_split[0]))

        delta_times = []
        for i in times:
            for j in times:
                delta_times.append(i-j)
        delta_times = list(set(delta_times))

        with open(os.path.join(self.data_path, 'delta_time.txt'), 'w') as f:
            for delta_time in delta_times:
                f.write(str(delta_time) + '\n')

        return



    def load_cluster(self):

        ent2clusterFile = os.path.join(self.data_path, 'dim50/entity2cluster.pkl')
        entity2cluster = pickle.load(open(ent2clusterFile, 'rb'))
        # with open(ent2clusterFile, 'r') as f:
        #     for line in f:
        #         line_split = line.split()  # entity cluster_label
        #         entity2cluster[int(line_split[0])] = int(line_split[1])

        rel2clusterFile = os.path.join(self.data_path, 'dim50/relation2cluster.pkl')
        relation2cluster = pickle.load(open(rel2clusterFile, 'rb'))
        # with open(rel2clusterFile, 'r') as f:
        #     for line in f:
        #         line_split = line.split()  # relation cluster_label
        #         relation2cluster[int(line_split[0])] = int(line_split[1])
        time2clusterFile = os.path.join(self.data_path, 'time2cluster.pkl')
        time2cluster = pickle.load(open(time2clusterFile, 'rb'))

        return entity2cluster, relation2cluster, time2cluster

    def product_triple_file(self):

        dataPath = '/home/luoxuewei/Projects/titer_-time-traveler-master_hrl_KM/data/YAGO/'
        train_entities =[]
        relations =[]
        triple_datas = []
        times = []
        with open(self.train_data_path, "r") as f:
            for line in f:
                line_split = line.split()
                head_time = int(line_split[0])
                tail_time = int(line_split[2])
                relations.append(int((line_split[1])))
                train_entities.append(head_time)
                train_entities.append(tail_time)
                triple_datas.append([int(line_split[0]), int(line_split[1]), int(line_split[2])])
                triple_datas.append \
                    ([int(line_split[2]), int(line_split[1] ) +self.real_rel_num +1, int(line_split[0])])
                times.append(int(line_split[3]))
        triple_datas.append([self.ePAD, self.real_rel_num, self.ePAD])
        triple_datas.append([self.ePAD, self.real_rel_num * 2 +1, self.ePAD])

        with open(self.valid_data_path, "r") as f:
            for line in f:
                line_split = line.split()  # head rel tail time
                head_time = (int(line_split[0]), int(line_split[3]))
                tail_time = (int(line_split[2]), int(line_split[3]))
                relations.append(int((line_split[1])))
                train_entities.append(head_time)
                train_entities.append(tail_time)
                times.append(int(line_split[3]))
                triple_datas.append([int(line_split[0]), int(line_split[1]), int(line_split[2])])
                triple_datas.append \
                    ([int(line_split[2]), int(line_split[1]) + self.real_rel_num + 1, int(line_split[0])])

        with open(self.test_data_path, "r") as f:
            for line in f:
                line_split = line.split()  # head rel tail time
                head_time = (int(line_split[0]), int(line_split[3]))
                tail_time = (int(line_split[2]), int(line_split[3]))
                relations.append(int((line_split[1])))
                train_entities.append(head_time)
                train_entities.append(tail_time)
                times.append(int(line_split[3]))
                triple_datas.append([int(line_split[0]), int(line_split[1]), int(line_split[2])])
                triple_datas.append \
                    ([int(line_split[2]), int(line_split[1]) + self.real_rel_num + 1, int(line_split[0])])


        times = list(set(times))
        time2clusterFile = os.path.join(dataPath, 'times.txt')
        with open(time2clusterFile, 'w') as f:
            for line in times:
                f.write(str(line) + '\n')



        '''emb_file_product'''
        ent2clusterFile = os.path.join(dataPath, 'Tucker/train.txt')
        with open(ent2clusterFile, 'w') as f:
            for line in triple_datas:
                f.write(str(line[0]) + '\t' + str(line[1]) + '\t' + str(line[2]) + '\n')
        #
        #
        train_entities = []
        relations = []
        triple_datas = []
        with open(self.valid_data_path, "r") as f:
            for line in f:
                line_split = line.split()  # head rel tail time
                head_time = int(line_split[0])
                tail_time = int(line_split[2])
                relations.append(int((line_split[1])))
                train_entities.append(head_time)
                train_entities.append(tail_time)
                triple_datas.append([int(line_split[0]), int(line_split[1]), int(line_split[2])])
                triple_datas.append \
                    ([int(line_split[2]), int(line_split[1]) + self.real_rel_num + 1, int(line_split[0])])

        '''emb_file_product'''
        ent2clusterFile = os.path.join(dataPath, 'Tucker/valid.txt')
        with open(ent2clusterFile, 'w') as f:
            for line in triple_datas:
                f.write(str(line[0]) + '\t' + str(line[1]) + '\t' + str(line[2]) + '\n')

        train_entities = []
        relations = []
        triple_datas = []
        with open(self.test_data_path, "r") as f:
            for line in f:
                line_split = line.split()  # head rel tail time
                head_time = int(line_split[0])
                tail_time = int(line_split[2])
                relations.append(int((line_split[1])))
                train_entities.append(head_time)
                train_entities.append(tail_time)
                triple_datas.append([int(line_split[0]), int(line_split[1]), int(line_split[2])])
                triple_datas.append \
                    ([int(line_split[2]), int(line_split[1]) + self.real_rel_num + 1, int(line_split[0])])

        '''emb_file_product'''
        ent2clusterFile = os.path.join(dataPath, 'Tucker/test.txt')
        with open(ent2clusterFile, 'w') as f:
            for line in triple_datas:
                f.write(str(line[0]) + '\t' + str(line[1]) + '\t' + str(line[2]) + '\n')


        train_entities.append((self.ePAD, self.tPAD))
