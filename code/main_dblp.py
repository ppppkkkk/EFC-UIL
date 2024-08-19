# -*- coding: utf-8 -*-
import numpy as np
import json,pickle,time,os

import scipy
from sklearn.decomposition import PCA

from align import align_cca
from utils import dataset,get_sim,hit_precision
from multiprocessing import Pool
from functools import partial


def compute_mrr(sim_matrix):
    mrr_sum = 0.0
    num_queries = sim_matrix.shape[0]

    for i in range(num_queries):
        # 获取该行的索引和值
        row_data = sim_matrix.getrow(i)
        indices = row_data.indices
        data = row_data.data

        # 补全缺失的列索引和相似度值
        missing_indices = set(range(num_queries)) - set(indices)
        indices = list(indices) + list(missing_indices)
        data = list(data) + [0.0] * len(missing_indices)

        # 根据值对索引进行排序
        sorted_indices = [index for _, index in sorted(zip(data, indices), reverse=True)]

        # 获取对角线索引的排名
        rank = sorted_indices.index(i) + 1

        # 更新 MRR
        mrr_sum += 1.0 / rank

    return mrr_sum / num_queries




def psearch(n_train,emb,K,reg,seed):
    test = datasets.get('test',n=500,seed=seed)
    train = datasets.get('train',n=n_train,seed=seed)
    traindata = []
    for k,v in train:
        traindata.append([emb[k],emb[v]])
    traindata = np.array(traindata)

    testdata = []
    for k,v in test:
        testdata.append([emb[k],emb[v]])
    testdata = np.array(testdata)
    
    zx,zy=align_cca(traindata,testdata,K=K,reg=reg)
    
    sim_matrix = get_sim(zx,zy,top_k=10)
    score=[]
    for top_k in [1,5,10]:
        score_ = hit_precision(sim_matrix,top_k=top_k)
        score.append(score_)

    mrr = compute_mrr(sim_matrix)
    score.append(mrr)
    return score

anchors = dict(json.load(open('../data/dblp/anchors.txt','r')))
print(time.ctime(),'\t # of Anchors:',len(anchors))
g1,g2 = pickle.load(open('../data/dblp/networks','rb'))
print(time.ctime(),'\t Size of two networks:',len(g1),len(g2)) 
datasets = dataset(anchors)


if __name__ == '__main__':
    #多线程需要放在主函数中
    pool = Pool(min(16, os.cpu_count() - 2))
    result=[]
    for seed in range(3):
        d = 100

        fname = '../emb/emb_dblp_seed_{}_dim_{}'.format(seed,d)
        '''
        emb_c, emb_m, emb_w, emb_t, emb_s = pickle.load(open(fname, 'rb'))
        print((emb_c.shape,emb_m.shape, emb_w.shape, emb_t.shape, emb_s.shape))
        emb_attr = np.concatenate((emb_c,emb_m,emb_w,emb_t),axis=-1)
        emb_all = np.concatenate((emb_c,emb_m,emb_w,emb_t,emb_s),axis=-1)
        '''
        emb_c, emb_t, emb_s = pickle.load(open(fname, 'rb'))
        # emb_c, emb_s = pickle.load(open(fname, 'rb'))
        # emb_attr = np.concatenate((emb_c),axis=-1)
        # emb_all = np.concatenate((emb_c,emb_s),axis=-1)
        emb_attr = np.concatenate((emb_c,emb_t),axis=-1)
        emb_all = np.concatenate((emb_c,emb_t,emb_s),axis=-1)
        for model in [2]:
            n_train = 200
            emb = [emb_attr,emb_s,emb_all][model]
            model_name = ['MAUIL-a','MAUIL-s','MAUIL'][model]
            dim = emb.shape[-1]
            for K in [[0],[0],[80]][model]:
                for reg in [100,1000]:
                    score=[]
                    seed_ = list(range(10))
                    score_10 = pool.map(partial(psearch,n_train,emb,K,reg),seed_)
                    score_10 = np.array(score_10)
                    assert score_10.shape==(10,4)
                    score = np.mean(score_10,axis=0)
    
                    record = [seed,d,model_name,n_train,K,reg]+score.tolist()
                    result.append(record)
                    print(record)

    json.dump(result,open('result_MAUIL_dblp.txt','w'))