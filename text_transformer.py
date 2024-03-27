import numpy as np
from sentence_transformers import SentenceTransformer, util
import os
import argparse
import pickle

def resave_entity_name(data_path, dataset):

    entity_names = []
    relation_names = []
    if dataset == 'ICEWS14':
        with open(os.path.join(data_path+dataset, 'entity2id.txt'), "r") as f:
            for line in f:
                line_split = line.split('\t')   # entity id
                entity_names.append(int(line_split[0]))

        with open(os.path.join(data_path+dataset, 'relation2id.txt'), "r") as f:
            for line in f:
                line_split = line.split('\t')   # entity id
                relation_names.append(int(line_split[0]))

    if dataset == 'ICEWS18':
        with open(os.path.join(data_path+dataset, '/entity2id.txt'), "r") as f:
            for line in f:
                line_split = line.split('\t')   # entity id
                entity_names.append(int(line_split[0]))

        with open(os.path.join(data_path+dataset, '/relation2id.txt'), "r") as f:
            for line in f:
                line_split = line.split('\t')   # entity id
                relation_names.append(int(line_split[0]))

    if dataset == 'YAGO':
        with open(os.path.join(data_path+dataset, '/entity2id.txt'), "r") as f:
            for line in f:
                line_split = line.split('\t')   # entity id
                entity_names.append(int(line_split[0]))

        with open(os.path.join(data_path+dataset, '/relation2id.txt'), "r") as f:
            for line in f:
                line_split = line.split('\t')   # entity id
                relation_names.append(int(line_split[0]))

    if dataset == 'ICEWS05-15':
        with open(os.path.join(data_path + dataset, '/entity2id.txt'), "r") as f:
            for line in f:
                line_split = line.split('\t')  # entity id
                entity_names.append(int(line_split[0]))

        with open(os.path.join(data_path + dataset, '/relation2id.txt'), "r") as f:
            for line in f:
                line_split = line.split('\t')  # entity id
                relation_names.append(int(line_split[0]))

    return entity_names, relation_names

def get_embedding_similarity(entity_names, relation_names):

    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Encode all sentences
    entity_name_embeddings = model.encode(entity_names)
    relation_name_embeddings = model.encode(relation_names)

    pad = np.zeros((1, entity_name_embeddings.shape[1]), dtype=np.float32)
    entity_name_embeddings = np.concatenate((entity_name_embeddings, pad), axis=0)
    pad = np.zeros((1, relation_name_embeddings.shape[1]), dtype=np.float32)
    relation_name_embeddings = np.concatenate((relation_name_embeddings, pad), axis=0)

    relation_name_embeddings_inv = relation_name_embeddings[:, ::-1]
    relation_name_embeddings = np.concatenate((relation_name_embeddings, relation_name_embeddings_inv), axis=0)

    return entity_name_embeddings, relation_name_embeddings

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='text_transformer', usage='text_transformer.py [<args>] [-h | --help]')
    parser.add_argument('--data_path', default='data/ICEWS14/', type=str, help='Path to data.')
    args = parser.parse_args()
    print(args)
    # extract entity names from entity2id.txt
    entity_names = []
    relation_names = []
    with open(os.path.join(args.data_path, 'entity2id.txt'), "r") as f:
        for line in f:
            line_split = line.split('\t')  # entity id
            entity_names.append(line_split[0])

    with open(os.path.join(args.data_path, 'relation2id.txt'), "r") as f:
        for line in f:
            line_split = line.split('\t')  # relation id
            relation_names.append(line_split[0])

    entity_names_embedding, relation_names_embedding = get_embedding_similarity(entity_names, relation_names)

    pickle.dump(entity_names_embedding, open(os.path.join(args.data_path, 'ent_text_emb.txt'), 'wb'))
    pickle.dump(relation_names_embedding, open(os.path.join(args.data_path, 'rel_text_emb.txt'), 'wb'))
