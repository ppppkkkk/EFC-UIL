# import numpy as np
# import json, pickle, time, os
# from multiprocessing import Pool
# from functools import partial
# from align import align_cca
# from utils import dataset, get_sim, hit_precision
#
#
# # 计算MRR
# def compute_mrr(sim_matrix):
#     mrr_sum = 0.0
#     num_queries = sim_matrix.shape[0]
#
#     for i in range(num_queries):
#         row_data = sim_matrix.getrow(i)
#         indices = row_data.indices
#         data = row_data.data
#
#         missing_indices = set(range(num_queries)) - set(indices)
#         indices = list(indices) + list(missing_indices)
#         data = list(data) + [0.0] * len(missing_indices)
#
#         sorted_indices = [index for _, index in sorted(zip(data, indices), reverse=True)]
#
#         rank = sorted_indices.index(i) + 1
#
#         mrr_sum += 1.0 / rank
#
#     return mrr_sum / num_queries
#
#
# anchors = dict(json.load(open('../data/dblp/anchors.txt', 'r')))
# print(time.ctime(), '\t # of Anchors:', len(anchors))
# g1, g2 = pickle.load(open('../data/dblp/networks', 'rb'))
# print(time.ctime(), '\t Size of two networks:', len(g1), len(g2))
# datasets = dataset(anchors)
#
#
# def psearch(emb, K, reg, train_ratio,seed):
#     # test = datasets.get('test', n=2587, seed=seed)
#     # train = datasets.get('train', n=1109, seed=seed)
#     # print(test)
#     # test = datasets.get('test', n=1982, seed=seed)
#     # train = datasets.get('train', n=850, seed=seed)
#     with open(f'train_anchors_dblp_2_{train_ratio}.pkl', 'rb') as f:
#         train = pickle.load(f)
#     with open(f'test_anchors_dblp_2_{train_ratio}.pkl', 'rb') as f:
#         test = pickle.load(f)
#
#     traindata = []
#     for k, v in train.items():  # 假设 train 是一个字典
#         traindata.append([emb[k], emb[v]])
#     traindata = np.array(traindata)
#
#     testdata = []
#     for k, v in test.items():  # 假设 test 是一个字典
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
#
#     return score
#
#
# if __name__ == '__main__':
#     pool = Pool(min(16, os.cpu_count() - 2))
#     result = []
#
#     for seed in [41]:
#         d = 768
#         emb_a, emb_s = pickle.load(open('../emb/emb_dblp_2_joint_initial', 'rb'))
#         emb_a = (emb_a - np.mean(emb_a, axis=0, keepdims=True)) / np.std(emb_a, axis=0, keepdims=True)
#         emb_s = (emb_s - np.mean(emb_s, axis=0, keepdims=True)) / np.std(emb_s, axis=0, keepdims=True)
#         print(emb_a.shape)
#         print(emb_s.shape)
#         emb_all = np.concatenate((emb_a, emb_s), axis=-1)
#
#         train_ratio = 0.7
#         for model_idx in [0,1,2]:
#             emb = [emb_a,emb_s,emb_all][model_idx]
#             model_name = ['EFC-UIL-a', 'EFC-UIL-s', 'EFC-UIL'][model_idx]
#             dim = emb.shape[-1]
#             for K in [[120], [120], [120]][model_idx]:
#                 for reg in [1000]:
#                     score = []
#                     seed_ = list(range(10))
#                     score_10 = pool.map(partial(psearch, emb, K, reg, train_ratio), seed_)
#                     score_10 = np.array(score_10)
#                     assert score_10.shape == (10, 4)
#                     score = np.mean(score_10, axis=0)
#                     score = np.round(score, 4)
#                     record = [seed, d, model_name, K, reg] + score.tolist()
#                     result.append(record)
#                     print(record)
#
#     json.dump(result, open('result_MAUIL_dblp.txt', 'w'))





# 定义门控网络（Router）
# class Router(nn.Module):
#     def __init__(self, input_dim, num_experts):
#         super(Router, self).__init__()
#         self.fc = nn.Linear(input_dim, num_experts)
#
#     def forward(self, x):
#         # 输出为每个专家的权重
#         weights = torch.softmax(self.fc(x), dim=-1)
#         return weights
#
#
# class MoELayer(nn.Module):
#     def __init__(self, input_dim, hidden_dim, output_dim, num_experts):
#         super(MoELayer, self).__init__()
#         self.experts = nn.ModuleList([FFN(input_dim, hidden_dim, output_dim) for _ in range(num_experts)])
#         self.router = Router(input_dim, num_experts)
#
#     def forward(self, x):
#         # 计算每个专家的权重得分
#         z = self.router(x)
#         # 通过softmax将得分归一化为选择权重
#         weights = torch.softmax(z, dim=-1)
#         # print(f"Normalized weights: {weights.shape}")
#         expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=-1)
#         # print(f"Expert outputs: {expert_outputs.shape}")
#         output = torch.sum(expert_outputs * weights.unsqueeze(1), dim=-1)
#         # print(f"Final weighted output: {output.shape}")
#
#         return output
#
#
# class TransformerWithMoE(nn.Module):
#     def __init__(self, input_dim, hidden_dim, output_dim, num_experts, num_layers):
#         super(TransformerWithMoE, self).__init__()
#         self.layers = nn.ModuleList([
#             nn.Sequential(
#                 nn.MultiheadAttention(embed_dim=input_dim, num_heads=8),
#                 nn.LayerNorm(input_dim),
#                 MoELayer(input_dim * 2, hidden_dim, output_dim, num_experts),
#                 nn.LayerNorm(output_dim)
#             )
#             for _ in range(num_layers)
#         ])
#
#     def forward(self, x_m, x_s):
#         for layer in self.layers:
#             # 分别对 x_m 和 x_s 进行注意力机制
#             attn_output_m, _ = layer[0](x_m, x_m, x_m)
#             attn_output_s, _ = layer[0](x_s, x_s, x_s)
#
#             # Add & Norm
#             x_m = layer[1](attn_output_m + x_m)
#             x_s = layer[1](attn_output_s + x_s)
#
#             combined_input = torch.cat([x_m, x_s], dim=-1)
#
#             moe_output = layer[2](combined_input)
#
#             final_output = torch.cat([moe_output, combined_input], dim=-1)
#
#         return final_output
import numpy as np
import json, pickle, time, os
from multiprocessing import Pool
from functools import partial
import torch
import torch.nn as nn
import torch.optim as optim
from align import align_cca
from utils import dataset, get_sim, hit_precision
import random


class FFN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FFN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class Router(nn.Module):
    def __init__(self, input_dim, num_experts):
        super(Router, self).__init__()
        self.fc_a = nn.Linear(input_dim, num_experts)  # 用于 emb_a 的权重
        self.fc_s = nn.Linear(input_dim, num_experts)  # 用于 emb_s 的权重

    def forward(self, x_m, x_s):
        # 分别计算 emb_a 和 emb_s 的专家权重
        weights_a = torch.softmax(self.fc_a(x_m), dim=-1)
        weights_s = torch.softmax(self.fc_s(x_s), dim=-1)
        return weights_a, weights_s


class MoELayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_experts):
        super(MoELayer, self).__init__()
        self.experts = nn.ModuleList([FFN(input_dim, hidden_dim, output_dim) for _ in range(num_experts)])
        self.router = Router(input_dim, num_experts)

    def forward(self, x_m, x_s):

        weights_a, weights_s = self.router(x_m, x_s)

        expert_outputs_a = torch.stack([expert(x_m) for expert in self.experts], dim=-1)
        expert_outputs_s = torch.stack([expert(x_s) for expert in self.experts], dim=-1)

        output_a = torch.sum(expert_outputs_a * weights_a.unsqueeze(1), dim=-1)
        output_s = torch.sum(expert_outputs_s * weights_s.unsqueeze(1), dim=-1)

        final_output = torch.cat([output_a, output_s], dim=-1)
        return final_output


class TransformerWithMoE(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_experts, num_layers):
        super(TransformerWithMoE, self).__init__()
        self.layers = nn.ModuleList([
            nn.Sequential(
                # nn.MultiheadAttention(embed_dim=input_dim, num_heads=8),
                nn.LayerNorm(input_dim),
                MoELayer(input_dim, hidden_dim, output_dim, num_experts),
                nn.LayerNorm(output_dim * 2)
            )
            for _ in range(num_layers)
        ])

    def forward(self, x_m, x_s):
        for layer in self.layers:
            # 注意力机制
            # attn_output_m, _ = layer[0](x_m, x_m, x_m)
            # attn_output_s, _ = layer[0](x_s, x_s, x_s)

            # Add & Norm
            x_m = layer[0](x_m)
            x_s = layer[0](x_s)

            combined_input = torch.cat([x_m, x_s], dim=-1)
            moe_output = layer[1](x_m, x_s)

            # final_output = torch.cat([moe_output, combined_input], dim=-1)
            # final_output = moe_output + combined_input
            final_output = layer[2](combined_input + moe_output)

        return final_output


def train_model(attribute_pairs, structure_pairs, model, num_epochs=10, learning_rate=0.001, batch_size=32,
                device='cpu'):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)

    clip_value = 1.0
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        for i in range(0, len(attribute_pairs), batch_size):
            batch_attr_pairs = attribute_pairs[i:i + batch_size]
            batch_struct_pairs = structure_pairs[i:i + batch_size]

            batch_attr_pairs = torch.tensor(batch_attr_pairs, dtype=torch.float32).to(device)
            batch_struct_pairs = torch.tensor(batch_struct_pairs, dtype=torch.float32).to(device)

            emb_k_m = batch_attr_pairs[:, 0, :]
            emb_v_m = batch_attr_pairs[:, 1, :]
            emb_k_s = batch_struct_pairs[:, 0, :]
            emb_v_s = batch_struct_pairs[:, 1, :]

            output_k = model(emb_k_m, emb_k_s)
            output_v = model(emb_v_m, emb_v_s)

            # # 打印当前 batch 的权重
            # print(f"Epoch {epoch+1} - Batch {i//batch_size+1}")
            # print(f"Weights for emb_k_m: {weights_a_k}")
            # print(f"Weights for emb_k_s: {weights_s_k}")
            # print(f"Weights for emb_v_m: {weights_a_v}")
            # print(f"Weights for emb_v_s: {weights_s_v}")

            l2_loss = torch.norm(output_k - output_v, p=2, dim=-1)
            loss = torch.mean(l2_loss)

            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)

            total_loss += loss.item()
            optimizer.step()

        # 更新学习率调度器
        scheduler.step(total_loss / len(attribute_pairs))

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / len(attribute_pairs):.4f}')


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


def psearch(emb_a, emb_s, K, reg, dataset, trained_model, train_ratio):
    moe_model = trained_model
    moe_model.eval()
    with open(f'train_anchors_{dataset}_{train_ratio}.pkl', 'rb') as f:
        train = pickle.load(f)
    with open(f'test_anchors_{dataset}_{train_ratio}.pkl', 'rb') as f:
        test = pickle.load(f)

    traindata = []
    for k, v in train.items():
        emb_k_m = torch.tensor(emb_a[k], dtype=torch.float32).unsqueeze(0)
        emb_k_s = torch.tensor(emb_s[k], dtype=torch.float32).unsqueeze(0)
        emb_v_m = torch.tensor(emb_a[v], dtype=torch.float32).unsqueeze(0)
        emb_v_s = torch.tensor(emb_s[v], dtype=torch.float32).unsqueeze(0)

        emb_k = moe_model(emb_k_m, emb_k_s).detach().numpy()
        emb_v = moe_model(emb_v_m, emb_v_s).detach().numpy()

        emb_k = emb_k.squeeze()
        emb_v = emb_v.squeeze()

        traindata.append([emb_k, emb_v])
    traindata = np.array(traindata)

    testdata = []
    for k, v in test.items():
        emb_k_m = torch.tensor(emb_a[k], dtype=torch.float32).unsqueeze(0)
        emb_k_s = torch.tensor(emb_s[k], dtype=torch.float32).unsqueeze(0)
        emb_v_m = torch.tensor(emb_a[v], dtype=torch.float32).unsqueeze(0)
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


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def psearch_with_seed(seed, emb_a, emb_s, K, reg, dataset, trained_model, train_ratio):
    set_seed(seed)
    return psearch(emb_a, emb_s, K, reg, dataset, trained_model, train_ratio)


def generate_pairs(anchors, emb_a, emb_s):
    attribute_pairs = []
    structure_pairs = []

    for node_a, node_b in anchors.items():
        emb_a_node_a = emb_a[int(node_a)]
        emb_a_node_b = emb_a[int(node_b)]

        emb_s_node_a = emb_s[int(node_a)]
        emb_s_node_b = emb_s[int(node_b)]

        attribute_pairs.append((emb_a_node_a, emb_a_node_b))
        structure_pairs.append((emb_s_node_a, emb_s_node_b))

    return np.array(attribute_pairs), np.array(structure_pairs)


anchors = dict(json.load(open('../data/dblp/anchors.txt', 'r')))
print(time.ctime(), '\t # of Anchors:', len(anchors))
g1, g2 = pickle.load(open('../data/dblp/networks', 'rb'))
print(time.ctime(), '\t Size of two networks:', len(g1), len(g2))
datasets = dataset(anchors)

if __name__ == '__main__':
    seed_value = 42
    set_seed(seed_value)
    train_ratio = 0.7
    pool = Pool(min(16, os.cpu_count() - 2))
    result = []
    dataset = 'dblp_1'
    for seed in [seed_value]:
        d = 768  # 输入维度
        emb_a, emb_s = pickle.load(open(f'../emb/emb_{dataset}_joint_initial', 'rb'))
        emb_a = (emb_a - np.mean(emb_a, axis=0, keepdims=True)) / np.std(emb_a, axis=0, keepdims=True)
        emb_s = (emb_s - np.mean(emb_s, axis=0, keepdims=True)) / np.std(emb_s, axis=0, keepdims=True)
        print(emb_a.shape)
        print(emb_s.shape)
        emb_all = np.concatenate((emb_a, emb_s), axis=-1)
        model = TransformerWithMoE(input_dim=768, hidden_dim=512, output_dim=768, num_experts=2, num_layers=1)
        print(f"Model initialized: {type(model)}")
        with open(f'train_anchors_{dataset}_{train_ratio}.pkl', 'rb') as f:
            train_anchors = pickle.load(f)

        set_seed(seed_value)
        attribute_pairs, structure_pairs = generate_pairs(train_anchors, emb_a, emb_s)

        print(f"Attribute pairs shape: {attribute_pairs.shape}")
        print(f"Structure pairs shape: {structure_pairs.shape}")
        # 训练模型
        train_model(attribute_pairs, structure_pairs, model, num_epochs=20, learning_rate=0.000001)

        # 对模型进行测试和评估
        for model_idx in [0]:
            model_name = ['EFC-UIL'][model_idx]
            dim = 768
            for K in [[120]][model_idx]:
                for reg in [1000]:
                    score = []
                    seed_ = list(range(10))
                    psearch_partial = partial(psearch_with_seed, emb_a=emb_a, emb_s=emb_s, K=K, reg=reg, dataset=dataset,
                                              trained_model=model, train_ratio=train_ratio)
                    score_10 = pool.map(psearch_partial, seed_)
                    score_10 = np.array(score_10)
                    assert score_10.shape == (10, 4)
                    score = np.mean(score_10, axis=0)

                    score = np.round(score, 4)

                    record = [seed, d, model_name, K, reg] + score.tolist()
                    result.append(record)
                    print(record)

    # 将结果保存到文件中，保留4位小数
    json.dump(result, open('result_EFC-UIL_dblp.txt', 'w'), indent=4)