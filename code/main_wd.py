import numpy as np
import json, pickle, time, os
from multiprocessing import Pool
from functools import partial
from align import align_cca
from utils import dataset, get_sim, hit_precision


# 计算MRR
def compute_mrr(sim_matrix):
    mrr_sum = 0.0
    num_queries = sim_matrix.shape[0]

    for i in range(num_queries):
        row_data = sim_matrix.getrow(i)
        indices = row_data.indices
        data = row_data.data

        missing_indices = set(range(num_queries)) - set(indices)
        indices = list(indices) + list(missing_indices)
        data = list(data) + [0.0] * len(missing_indices)

        sorted_indices = [index for _, index in sorted(zip(data, indices), reverse=True)]

        rank = sorted_indices.index(i) + 1

        mrr_sum += 1.0 / rank

    return mrr_sum / num_queries


anchors = dict(json.load(open('../data/wd/anchors.txt', 'r')))
print(time.ctime(), '\t # of Anchors:', len(anchors))
g1, g2 = pickle.load(open('../data/wd/networks', 'rb'))
print(time.ctime(), '\t Size of two networks:', len(g1), len(g2))
datasets = dataset(anchors)


def psearch(emb, K, reg, seed):
    test = datasets.get('test', n=978, seed=seed)
    train = datasets.get('train', n=319, seed=seed)
    traindata = []
    for k, v in train:
        traindata.append([emb[k], emb[v]])
    traindata = np.array(traindata)
    testdata = []
    for k, v in test:
        testdata.append([emb[k], emb[v]])
    testdata = np.array(testdata)

    zx, zy = align_cca(traindata, testdata, K=K, reg=reg)

    sim_matrix = get_sim(zx, zy, top_k=10)
    score = []
    for top_k in [1, 5, 10]:
        score_ = hit_precision(sim_matrix, top_k=top_k)
        score.append(score_)

    mrr = compute_mrr(sim_matrix)
    score.append(mrr)
    return score


if __name__ == '__main__':
    pool = Pool(min(16, os.cpu_count() - 2))
    result = []

    for seed in [41]:
        d = 768
        emb_m, emb_s = pickle.load(open('../emb/emb_wd_joint_initial', 'rb'))
        emb_m = pickle.load(open('../emb/bert_embeddings_wd.pkl', 'rb'))
        emb_all = np.concatenate((emb_m, emb_s), axis=-1)


        for model_idx in [0,1,2]:
            emb = [emb_m,emb_s,emb_all][model_idx]
            model_name = ['EFC-UIL-a', 'EFC-UIL-s', 'EFC-UIL'][model_idx]
            dim = emb.shape[-1]
            for K in [[120], [120], [120]][model_idx]:
                for reg in [100, 1000]:
                    score = []
                    seed_ = list(range(10))
                    score_10 = pool.map(partial(psearch, emb, K, reg), seed_)
                    score_10 = np.array(score_10)
                    assert score_10.shape == (10, 4)
                    score = np.mean(score_10, axis=0)
                    score = np.round(score, 4)
                    record = [seed, d, model_name, K, reg] + score.tolist()
                    result.append(record)
                    print(record)

    json.dump(result, open('result_EFC-UIL_dblp.txt', 'w'))