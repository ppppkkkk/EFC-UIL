import numpy as np
import json, pickle, time, os
from multiprocessing import Pool
from functools import partial

import torch
import torch.nn as nn
import torch.optim as optim

# 从你的现有模块导入相关函数
from align import align_cca
from utils import dataset, get_sim, hit_precision

# 定义一个全局的前馈神经网络门控网络
class GlobalGatingNetwork(nn.Module):
    def __init__(self):
        super(GlobalGatingNetwork, self).__init__()
        # 初始化为可训练的全局权重参数
        self.weights = nn.Parameter(torch.tensor([0.5, 0.5]), requires_grad=True)

    def forward(self):
        # 使用softmax确保权重和为1
        weights = torch.softmax(self.weights, dim=-1)
        return weights

# 定义带GlobalGatingNetwork的混合专家模型
class MixtureOfExpertsWithGlobalGating(nn.Module):
    def __init__(self, emb_m_dim, emb_s_dim):
        super(MixtureOfExpertsWithGlobalGating, self).__init__()
        self.gating_network = GlobalGatingNetwork()

    def forward(self, emb_m, emb_s):
        # 计算全局权重
        weights = self.gating_network()

        # 对所有嵌入应用相同的权重
        output = emb_m * weights[0] + emb_s * weights[1]
        return output


# 生成属性和结构目标对
# 生成属性和结构目标对
def generate_target(emb_m, emb_s, anchors):
    attribute_pairs = []
    structure_pairs = []
    labels = []

    for k, v in anchors.items():
        emb_k_m = emb_m[k]
        emb_v_m = emb_m[v]
        emb_k_s = emb_s[k]
        emb_v_s = emb_s[v]

        attribute_pairs.append([emb_k_m, emb_v_m])
        structure_pairs.append([emb_k_s, emb_v_s])
        labels.append(1)

    return np.array(attribute_pairs), np.array(structure_pairs), np.array(labels)


def train_model(attribute_pairs, structure_pairs, labels, model, num_epochs=10, learning_rate=0.001):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0  # 初始化为浮点数

        for (attr_pair, struct_pair, label) in zip(attribute_pairs, structure_pairs, labels):
            emb_k_m = torch.tensor(attr_pair[0], dtype=torch.float32).unsqueeze(0)
            emb_v_m = torch.tensor(attr_pair[1], dtype=torch.float32).unsqueeze(0)
            emb_k_s = torch.tensor(struct_pair[0], dtype=torch.float32).unsqueeze(0)
            emb_v_s = torch.tensor(struct_pair[1], dtype=torch.float32).unsqueeze(0)

            aligned_emb_k = torch.cat([emb_k_m, emb_k_s], dim=-1)
            aligned_emb_v = torch.cat([emb_v_m, emb_v_s], dim=-1)

            output_k = model(aligned_emb_k[:, :emb_k_m.size(-1)], aligned_emb_k[:, emb_k_m.size(-1):])
            output_v = model(aligned_emb_v[:, :emb_v_m.size(-1)], aligned_emb_v[:, emb_v_m.size(-1):])

            logits = torch.cosine_similarity(output_k, output_v, dim=-1)
            logits = logits.view(-1)

            label_tensor = torch.tensor([label], dtype=torch.float32)

            loss = criterion(logits, label_tensor)

            optimizer.zero_grad()
            total_loss += loss.item()  # 累加损失
            loss.backward()
            optimizer.step()

        # 打印平均损失
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(attribute_pairs):.4f}')

    # 打印最终的全局权重
    final_weights = model.gating_network().detach().numpy()
    print(f"Final global weights after training: {final_weights}")


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

#定义psearch函数
def psearch(emb_m, emb_s, K, reg, seed, trained_model):
    moe_model = trained_model
    moe_model.eval()  # 设置模型为评估模式

    test = datasets.get('test', n=2000, seed=seed)
    train = datasets.get('train', n=850, seed=seed)

    traindata = []
    for k, v in train:
        emb_k_m = torch.tensor(emb_m[k], dtype=torch.float32).unsqueeze(0)
        emb_k_s = torch.tensor(emb_s[k], dtype=torch.float32).unsqueeze(0)
        emb_v_m = torch.tensor(emb_m[v], dtype=torch.float32).unsqueeze(0)
        emb_v_s = torch.tensor(emb_s[v], dtype=torch.float32).unsqueeze(0)

        emb_k = moe_model(emb_k_m, emb_k_s).detach().numpy()
        emb_v = moe_model(emb_v_m, emb_v_s).detach().numpy()

        emb_k = emb_k.squeeze()
        emb_v = emb_v.squeeze()

        traindata.append([emb_k, emb_v])
    traindata = np.array(traindata)

    testdata = []
    for k, v in test:
        emb_k_m = torch.tensor(emb_m[k], dtype=torch.float32).unsqueeze(0)
        emb_k_s = torch.tensor(emb_s[k], dtype=torch.float32).unsqueeze(0)
        emb_v_m = torch.tensor(emb_m[v], dtype=torch.float32).unsqueeze(0)
        emb_v_s = torch.tensor(emb_s[v], dtype=torch.float32).unsqueeze(0)

        emb_k = moe_model(emb_k_m, emb_k_s).detach().numpy()
        emb_v = moe_model(emb_v_m, emb_v_s).detach().numpy()

        emb_k = emb_k.squeeze()
        emb_v = emb_v.squeeze()

        testdata.append([emb_k, emb_v])
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


anchors = dict(json.load(open('../data/dblp/anchors.txt', 'r')))
print(time.ctime(), '\t # of Anchors:', len(anchors))
g1, g2 = pickle.load(open('../data/dblp/networks', 'rb'))
print(time.ctime(), '\t Size of two networks:', len(g1), len(g2))
datasets = dataset(anchors)
#
# def psearch(emb, K, reg, seed):
#     test = datasets.get('test', n=2000, seed=seed)
#     train = datasets.get('train', n=850, seed=seed)
#     traindata = []
#     for k, v in train:
#         traindata.append([emb[k], emb[v]])
#     traindata = np.array(traindata)
#
#     testdata = []
#     for k, v in test:
#         testdata.append([emb[k], emb[v]])
#     testdata = np.array(testdata)
#
#     zx, zy = align_cca(traindata, testdata, K=K, reg=reg)
#
#     sim_matrix = get_sim(zx, zy, top_k=10)
#     score = []
#     for top_k in [1, 5, 10]:
#         score_ = hit_precision(sim_matrix, top_k=top_k)
#         score.append(score_)
#
#     mrr = compute_mrr(sim_matrix)
#     score.append(mrr)
#     return score



if __name__ == '__main__':
    pool = Pool(min(16, os.cpu_count() - 2))
    result = []

    for seed in range(1):
        d = 768
        emb_m, emb_s = pickle.load(open('../emb/emb_dblp_seed_0_dim_768', 'rb'))
        emb_t = pickle.load(open('../emb/emb_dblp1_dim', 'rb'))
        print(emb_m)
        print(f"emb_m shape: {emb_m.shape}")
        print(f"emb_s shape: {emb_s.shape}")
        emb_all = np.concatenate((emb_m, emb_s), axis=-1)

        #生成目标对
        attribute_pairs, structure_pairs, labels = generate_target(emb_m, emb_s, anchors)

        # 初始化使用GlobalGatingNetwork的模型
        model = MixtureOfExpertsWithGlobalGating(emb_m_dim=768, emb_s_dim=768)
        print(f"Model initialized: {type(model)}")

        # 训练模型
        train_model(attribute_pairs, structure_pairs, labels,  model, num_epochs=10, learning_rate=0.001)

        for model_idx in [0]:
            emb = [emb_m, emb_s, emb_all][model_idx]
            model_name = ['MAUIL-a', 'MAUIL-s', 'MAUIL'][model_idx]
            dim = emb.shape[-1]
            for K in [[120], [120], [120]][model_idx]:
                for reg in [100, 1000]:
                    score = []
                    seed_ = list(range(10))
                    psearch_partial = partial(psearch, emb_m, emb_s, K, reg, trained_model=model)
                    score_10 = pool.map(psearch_partial, seed_)
                    #score_10 = pool.map(partial(psearch, emb, K, reg), seed_)
                    score_10 = np.array(score_10)
                    assert score_10.shape == (10, 4)
                    score = np.mean(score_10, axis=0)

                    record = [seed, d, model_name, K, reg] + score.tolist()
                    result.append(record)
                    print(record)

    json.dump(result, open('result_MAUIL_dblp.txt', 'w'))
