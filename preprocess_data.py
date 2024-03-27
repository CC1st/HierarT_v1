import pickle
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import argparse
from model.environment import Env
from dataset.baseDataset import baseDataset
from tqdm import tqdm
import torch
from sklearn.cluster import KMeans

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Data preprocess', usage='preprocess_data.py [<args>] [-h | --help]')
    parser.add_argument('--data_dir', default='data/ICEWS14', type=str, help='Path to data.')
    parser.add_argument('--outfile', default='state_actions_space.pkl', type=str,
                        help='file to save the preprocessed data.')
    parser.add_argument('--store_actions_num', default=None, type=int,
                        help='maximum number of stored neighbors, 0 means store all.')
    parser.add_argument('--store_options_num', default=None, type=int,
                        help='maximum number of stored neighbors, 0 means store all.')
    parser.add_argument('--dim_t',default=20, type=int, help='the dimension of time embeddimg')
    args = parser.parse_args()

    trainF = os.path.join(args.data_dir, 'train.txt')
    testF = os.path.join(args.data_dir, 'test.txt')
    statF = os.path.join(args.data_dir, 'stat.txt')
    validF = os.path.join(args.data_dir, 'valid.txt')
    if not os.path.exists(validF):
        validF = None
    dataset = baseDataset(trainF, testF, statF, validF)
    path_split = args.data_dir.split('/')
    dataset_name = path_split[-1]
    config = {
        'num_rel': dataset.num_r,
        'num_ent': dataset.num_e,
        'num_t': dataset.num_t,
        'dataset': dataset_name,
    }
    env = Env(dataset.allQuadruples, config)
    state_actions_space = {}
    state_actions_space_r_et = {}
    timestamps = list(dataset.get_all_timestamps())
    print(args)
    abst_embeddimg = env.abst_embds_productor(dataset.num_t+1, args.dim_t)
    pickle.dump(abst_embeddimg, open(os.path.join(args.data_dir+'/Tucker/', 'abst2emb.pkl'), 'wb'))
    fact_num = 0
    with tqdm(total=len(dataset.allQuadruples)) as bar:
        for (head, rel, tail, t) in dataset.allQuadruples:
            if (head, t, True) not in state_actions_space.keys():
                state_actions_space[(head, t, True)] = env.get_state_actions_space_complete_predata(head, t, True, args.store_options_num, args.store_actions_num)
                state_actions_space[(head, t, False)] = env.get_state_actions_space_complete_predata(head, t, False, args.store_options_num, args.store_actions_num)

            if (tail, t, True) not in state_actions_space.keys():
                state_actions_space[(tail, t, True)] = env.get_state_actions_space_complete_predata(tail, t, True, args.store_options_num, args.store_actions_num)
                state_actions_space[(tail, t, False)] = env.get_state_actions_space_complete_predata(tail, t, False, args.store_options_num, args.store_actions_num)

            bar.update(1)

    pickle.dump(state_actions_space, open(os.path.join(args.data_dir, args.outfile), 'wb'))

