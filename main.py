#-- coding:UTF-8 --

import argparse
# import numpy as np
import torch
from torch.utils.data import DataLoader
from utils.logger import *
from utils.trainer import Trainer
from utils.tester import Tester
from dataset.baseDataset import baseDataset, QuadruplesDataset
# from model.agent import Agent
from model.agent import High_level_Agent
from model.agent import Low_level_Agent
from model.environment import Env
from model.episode import Episode
from model.policyGradient import PG
from model.dirichlet import Dirichlet
from model.KMeans import Ent2Cluster
import os
import pickle

def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Training and Testing Temporal Knowledge Graph Forecasting Models',
        usage='main.py [<args>] [-h | --help]'
    )

    parser.add_argument('--cuda', action='store_true', help='whether to use GPU or not.')
    parser.add_argument('--device_id', default=0, type=int, help='the CUDA number of the running program')
    parser.add_argument('--data_path', type=str, default='data/ICEWS14', help='Path to data.')
    parser.add_argument('--do_train', action='store_true', help='whether to train.')
    parser.add_argument('--do_test', action='store_true', help='whether to test.')
    parser.add_argument('--save_path', default='logs', type=str, help='log and model save path.')
    parser.add_argument('--load_model_path', default='logs', type=str, help='trained model checkpoint path.')

    # Train Params
    parser.add_argument('--batch_size', default=512, type=int, help='training batch size.')
    parser.add_argument('--max_epochs', default=400, type=int, help='max training epochs.')
    parser.add_argument('--num_workers', default=8, type=int, help='workers number used for dataloader.')
    parser.add_argument('--valid_epoch', default=30, type=int, help='validation frequency.')
    parser.add_argument('--lr_hl', default=0.0003, type=float, help='learning rate of high-level.')
    parser.add_argument('--lr_ll', default=0.0001, type=float, help='learning rate of low-level.')
    parser.add_argument('--save_epoch', default=30, type=int, help='model saving frequency.')
    parser.add_argument('--clip_gradient', default=10.0, type=float, help='for gradient crop.')

    # Test Params
    parser.add_argument('--test_batch_size', default=1, type=int,
                        help='test batch size, it needs to be set to 1 when using IM module.')
    parser.add_argument('--beam_size', default=80, type=int, help='the beam number of the beam search.')
    parser.add_argument('--hl_beam_size', default=10, type=int, help='the beam number of the high level beam search.')
    parser.add_argument('--ll_beam_size', default=60, type=int, help='the beam number of the low level beam search.')
    parser.add_argument('--test_inductive', action='store_true', help='whether to verify inductive inference performance.')
    parser.add_argument('--IM', action='store_true', help='whether to use IM module.')
    parser.add_argument('--mu', default=0.1, type=float, help='the hyperparameter of IM module.')

    # Agent Params
    parser.add_argument('--ent_dim', default=100, type=int, help='Embedding dimension of the entities')
    parser.add_argument('--rel_dim', default=100, type=int, help='Embedding dimension of the relations')
    parser.add_argument('--state_dim', default=100, type=int, help='dimension of the LSTM hidden state')
    parser.add_argument('--hidden_dim', default=100, type=int, help='dimension of the MLP hidden layer')
    parser.add_argument('--time_dim', default=20, type=int, help='Embedding dimension of the timestamps')
    parser.add_argument('--entities_embeds_method', default='dynamic', type=str,
                        help='representation method of the entities, dynamic or static')
    parser.add_argument('--dp0_hl', default=0.3, type=float, help='Dropout rate of PolicyMLP_High')
    parser.add_argument('--dp1_hl', default=0.1, type=float, help='Dropout rate of PolicyMLP_High')
    parser.add_argument('--dp2_hl', default=0.1, type=float, help='Dropout rate of PolicyMLP_High')
    parser.add_argument('--dp0_ll', default=0.2, type=float, help='Dropout rate of PolicyMLP_Low')
    parser.add_argument('--dp1_ll', default=0.1, type=float, help='Dropout rate of PolicyMLP_Low')
    parser.add_argument('--dp2_ll', default=0.1, type=float, help='Dropout rate of PolicyMLP_Low')

    # Environment Params
    parser.add_argument('--state_actions_path', default='state_actions_space.pkl', type=str,
                        help='the file stores preprocessed candidate action array.')
    parser.add_argument('--state_actions_path_source', default='state_actions_space_source.pkl', type=str,
                        help='the file stores preprocessed candidate action array of tail entities.')
    parser.add_argument('--state_actions_space_re', default='state_actions_space_re.pkl', type=str,
                        help='the file stores preprocessed candidate action array of tail entities.')

    # Episode Params
    parser.add_argument('--path_length', default=2, type=int, help='the agent search path length.')
    parser.add_argument('--max_option_num', default=None, type=int, help='the max candidate actions number.')
    parser.add_argument('--max_action_num', default=None, type=int, help='the max candidate actions number.')

    # Policy Gradient Params
    parser.add_argument('--Lambda', default=0.0, type=float, help='update rate of baseline.')
    parser.add_argument('--Gamma', default=0.95, type=float, help='discount factor of Bellman Eq.')
    parser.add_argument('--Ita', default=0.01, type=float, help='regular proportionality constant.')
    parser.add_argument('--Zita', default=0.9, type=float, help='attenuation factor of entropy regular term.')

    # reward shaping params
    parser.add_argument('--reward_shaping', action='store_true', help='whether to use reward shaping.')
    parser.add_argument('--time_span', default=24, type=int, help='24 for ICEWS, 1 for WIKI and YAGO')
    parser.add_argument('--alphas_pkl', default='dirchlet_alphas.pkl', type=str,
                        help='the file storing the alpha parameters of the Dirichlet distribution.')
    parser.add_argument('--k', default=300, type=int, help='statistics recent K historical snapshots.')

    # kmeans cluster
    parser.add_argument('--kmeans_reward_shaping', action='store_true', default=False, help='whether use the module of kmeans reward shaping')
    parser.add_argument('--train_entity_k', default=4, type=int, help='Number of KMeans clusters')
    parser.add_argument('--train_rel_k', default=5, type=int, help='Number of KMeans clusters')
    parser.add_argument('--time_k', default=6, type=int, help='Number of KMeans clusters')

    # words semantics transformer
    parser.add_argument('--text_transformer', action='store_true', default=False, help='whether use the module of text transformer')

    return parser.parse_args(args)

def get_model_config(args, num_ent, num_rel, num_t):
    path_split = args.data_path.split('/')
    dataset = path_split[-1]

    config = {
        'cuda': args.cuda,  # whether to use GPU or not.
        'batch_size': args.batch_size,  # training batch size.
        'num_ent': num_ent,  # number of entities
        'num_rel': num_rel,  # number of relations
        'num_t': num_t,
        'ent_dim': args.ent_dim,  # Embedding dimension of the entities
        'rel_dim': args.rel_dim,  # Embedding dimension of the relations
        'time_dim': args.time_dim,  # Embedding dimension of the timestamps
        'state_dim': args.state_dim,  # dimension of the LSTM hidden state
        'action_dim': args.ent_dim + args.rel_dim,  # dimension of the actions
        'mlp_input_dim': args.ent_dim + args.rel_dim + args.state_dim,  # dimension of the input of the MLP
        'mlp_input_dim_high': args.rel_dim + args.state_dim,  # dimension of the input of the MLP
        'mlp_input_dim_low': args.ent_dim + args.state_dim,  # dimension of the input of the MLP
        'mlp_hidden_dim': args.hidden_dim,  # dimension of the MLP hidden layer
        'path_length': args.path_length,  # agent search path length
        'max_option_num': args.max_option_num,
        'max_action_num': args.max_action_num,  # max candidate action number
        'lambda': args.Lambda,  # update rate of baseline
        'gamma': args.Gamma,  # discount factor of Bellman Eq.
        'ita': args.Ita,  # regular proportionality constant
        'zita': args.Zita,  # attenuation factor of entropy regular term
        'beam_size': args.beam_size,  # beam size for beam search
        'hl_beam_size': args.hl_beam_size,  # the beam number of the high level beam search.
        'll_beam_size': args.ll_beam_size,  # the beam number of the low level beam search.
        'entities_embeds_method': args.entities_embeds_method,  # default: 'dynamic', otherwise static encoder will be used
        'device_id': args.device_id,
        'train_data_path': os.path.join(args.data_path, 'train.txt'),
        'valid_data_path': os.path.join(args.data_path, 'valid.txt'),
        'test_data_path': os.path.join(args.data_path, 'test.txt'),
        'train_entity_k': args.train_entity_k,
        'train_rel_k': args.train_rel_k,
        'time_k': args.time_k,
        'data_path': args.data_path,
        'dp0_hl':args.dp0_hl,
        'dp1_hl':args.dp1_hl,
        'dp2_hl':args.dp2_hl,
        'dp0_ll':args.dp0_ll,
        'dp1_ll':args.dp1_ll,
        'dp2_ll':args.dp2_ll,
        'dataset':dataset,
    }
    return config

def main(args):
    #######################Set Logger#################################
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    if args.cuda and torch.cuda.is_available():
        args.cuda = True
        torch.cuda.set_device(args.device_id)
    else:
        args.cuda = False
    set_logger(args)

    #######################Create DataLoader#################################
    train_path = os.path.join(args.data_path, 'train.txt')
    test_path = os.path.join(args.data_path, 'test.txt')
    stat_path = os.path.join(args.data_path, 'stat.txt')
    valid_path = os.path.join(args.data_path, 'valid.txt')


    baseData = baseDataset(train_path, test_path, stat_path, valid_path)

    trainDataset  = QuadruplesDataset(baseData.trainQuadruples, baseData.num_r)
    train_dataloader = DataLoader(
        trainDataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )

    validDataset = QuadruplesDataset(baseData.validQuadruples, baseData.num_r)
    valid_dataloader = DataLoader(
        validDataset,
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    testDataset = QuadruplesDataset(baseData.testQuadruples, baseData.num_r)
    test_dataloader = DataLoader(
        testDataset,
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    ######################Creat the agent and the environment###########################
    config = get_model_config(args, baseData.num_e, baseData.num_r, baseData.num_t)
    logging.info(config)
    logging.info(args)

    # use kmeans reward shaping
    kmeans_cluster = None
    if args.kmeans_reward_shaping:
        kmeans_cluster = Ent2Cluster(config)

    # text transformer
    ent_text_embd = None
    rel_text_embd = None
    ent_cos_sim_path = os.path.join(args.data_path, 'ent_text_emb.txt')
    rel_cos_sim_path = os.path.join(args.data_path, 'rel_text_emb.txt')
    if args.text_transformer and os.path.exists(ent_cos_sim_path):
        ent_text_embd = pickle.load(open(ent_cos_sim_path, 'rb'))
        rel_text_embd = pickle.load(open(rel_cos_sim_path, 'rb'))
        ent_text_embd = torch.tensor(ent_text_embd).cuda()
        rel_text_embd = torch.tensor(rel_text_embd).cuda()


    # abst_embeddimg
    abst_embds_path = os.path.join(args.data_path+'/Tucker/', 'abst2emb.pkl')
    if os.path.exists(abst_embds_path):
        abst_embedding = pickle.load(open(abst_embds_path, 'rb'))
        abst_embedding = torch.tensor(abst_embedding, dtype=torch.float32).cuda()
    else:
        raise ValueError('not exit the path: %s' % abst_embds_path)

    agent_hl = High_level_Agent(config, rel_text_embd)
    agent_ll = Low_level_Agent(config, abst_embedding, ent_text_embd)

    # creat the environment
    state_actions_path = os.path.join(args.data_path, args.state_actions_path)

    if not os.path.exists(state_actions_path):
        state_action_space = None
    else:
        state_action_space = pickle.load(open(state_actions_path, 'rb'))

    env = Env(baseData.allQuadruples, config, state_action_space)

    # Create episode controller
    episode = Episode(env, agent_hl, agent_ll, config, kmeans_cluster)
    if args.cuda:
        episode = episode.cuda()
    pg = PG(config)  # Policy Gradient
    optimizer_hl = torch.optim.Adam(episode.agent_hl.parameters(), lr=args.lr_hl, weight_decay=0.00001)
    optimizer_ll = torch.optim.Adam(episode.agent_ll.parameters(), lr=args.lr_ll, weight_decay=0.00001)


    # Load the model parameters
    if os.path.isfile(args.load_model_path):
        params = torch.load(args.load_model_path)
        episode.load_state_dict(params['model_state_dict'])
        optimizer_hl.load_state_dict(params['optimizer_hl_state_dict'])
        optimizer_ll.load_state_dict(params['optimizer_ll_state_dict'])
        logging.info('Load pretrain model: {}'.format(args.load_model_path))

    ######################Training and Testing###########################
    if args.reward_shaping:
        alphas = pickle.load(open(os.path.join(args.data_path, args.alphas_pkl), 'rb'))
        distributions = Dirichlet(alphas, args.k)
    else:
        distributions = None
    trainer = Trainer(episode, pg, optimizer_hl, optimizer_ll, args, distributions)
    tester = Tester(episode, args, baseData.train_entities, baseData.RelEntCooccurrence)
    if args.do_train:
        logging.info('Start Training......')
        for i in range(args.max_epochs):
            loss, reward = trainer.train_epoch(train_dataloader, trainDataset.__len__())
            logging.info('Epoch {}/{} Loss: {}, reward: {}'.format(i, args.max_epochs, loss, reward))

            if i % args.save_epoch == 0 and i != 0:
                trainer.save_model('checkpoint_{}.pth'.format(i))
                logging.info('Save Model in {}'.format(args.save_path))

            if i % args.valid_epoch == 0 and i != 0:
                logging.info('Start Val......')
                metrics = tester.test(valid_dataloader,
                                      validDataset.__len__(),
                                      baseData.skip_dict,
                                      config['num_ent'])
                for mode in metrics.keys():
                    logging.info('{} at epoch {}: {}'.format(mode, i, metrics[mode]))

        trainer.save_model()
        logging.info('Save Model in {}'.format(args.save_path))

    if args.do_test:
        logging.info('Start Testing......')

        metrics = tester.test(test_dataloader,
                              testDataset.__len__(),
                              baseData.skip_dict,
                              config['num_ent'])
        for mode in metrics.keys():
            logging.info('Test {} : {}'.format(mode, metrics[mode]))

if __name__ == '__main__':
    args = parse_args()
    main(args)
