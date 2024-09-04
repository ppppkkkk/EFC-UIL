# import numpy as np
# import json, pickle, time, os
# from multiprocessing import Pool
# from functools import partial
#
# import torch
# import torch.nn as nn
# import torch.optim as optim
#
# # 从你的现有模块导入相关函数
# from align import align_cca
# from utils import dataset, get_sim, hit_precision
#
# # 定义一个全局的前馈神经网络门控网络
# class GlobalGatingNetwork(nn.Module):
#     def __init__(self):
#         super(GlobalGatingNetwork, self).__init__()
#         # 初始化为可训练的全局权重参数
#         self.weights = nn.Parameter(torch.tensor([0.5, 0.5]), requires_grad=True)
#
#     def forward(self):
#         # 使用softmax确保权重和为1
#         weights = torch.softmax(self.weights, dim=-1)
#         return weights
#
# # 定义带GlobalGatingNetwork的混合专家模型
# class MixtureOfExpertsWithGlobalGating(nn.Module):
#     def __init__(self, emb_m_dim, emb_s_dim):
#         super(MixtureOfExpertsWithGlobalGating, self).__init__()
#         self.gating_network = GlobalGatingNetwork()
#
#     def forward(self, emb_m, emb_s):
#         # 计算全局权重
#         weights = self.gating_network()
#
#         # 对所有嵌入应用相同的权重
#         output = emb_m * weights[0] + emb_s * weights[1]
#         return output
#
#
# # 生成属性和结构目标对
# # 生成属性和结构目标对
# def generate_target(emb_m, emb_s, anchors):
#     attribute_pairs = []
#     structure_pairs = []
#     labels = []
#
#     for k, v in anchors.items():
#         emb_k_m = emb_m[k]
#         emb_v_m = emb_m[v]
#         emb_k_s = emb_s[k]
#         emb_v_s = emb_s[v]
#
#         attribute_pairs.append([emb_k_m, emb_v_m])
#         structure_pairs.append([emb_k_s, emb_v_s])
#         labels.append(1)
#
#     return np.array(attribute_pairs), np.array(structure_pairs), np.array(labels)
#
#
# def train_model(attribute_pairs, structure_pairs, labels, model, num_epochs=10, learning_rate=0.001):
#     criterion = nn.BCEWithLogitsLoss()
#     optimizer = optim.Adam(model.parameters(), lr=learning_rate)
#
#     for epoch in range(num_epochs):
#         model.train()
#         total_loss = 0.0  # 初始化为浮点数
#
#         for (attr_pair, struct_pair, label) in zip(attribute_pairs, structure_pairs, labels):
#             emb_k_m = torch.tensor(attr_pair[0], dtype=torch.float32).unsqueeze(0)
#             emb_v_m = torch.tensor(attr_pair[1], dtype=torch.float32).unsqueeze(0)
#             emb_k_s = torch.tensor(struct_pair[0], dtype=torch.float32).unsqueeze(0)
#             emb_v_s = torch.tensor(struct_pair[1], dtype=torch.float32).unsqueeze(0)
#
#             aligned_emb_k = torch.cat([emb_k_m, emb_k_s], dim=-1)
#             aligned_emb_v = torch.cat([emb_v_m, emb_v_s], dim=-1)
#
#             output_k = model(aligned_emb_k[:, :emb_k_m.size(-1)], aligned_emb_k[:, emb_k_m.size(-1):])
#             output_v = model(aligned_emb_v[:, :emb_v_m.size(-1)], aligned_emb_v[:, emb_v_m.size(-1):])
#
#             logits = torch.cosine_similarity(output_k, output_v, dim=-1)
#             logits = logits.view(-1)
#
#             label_tensor = torch.tensor([label], dtype=torch.float32)
#
#             loss = criterion(logits, label_tensor)
#
#             optimizer.zero_grad()
#             total_loss += loss.item()  # 累加损失
#             loss.backward()
#             optimizer.step()
#
#         # 打印平均损失
#         print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(attribute_pairs):.4f}')
#
#     # 打印最终的全局权重
#     final_weights = model.gating_network().detach().numpy()
#     print(f"Final global weights after training: {final_weights}")
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
# # #定义psearch函数
# # def psearch(emb_m, emb_s, K, reg, seed, trained_model):
# #     moe_model = trained_model
# #     moe_model.eval()  # 设置模型为评估模式
# #
# #     test = datasets.get('test', n=2000, seed=seed)
# #     train = datasets.get('train', n=850, seed=seed)
# #
# #     traindata = []
# #     for k, v in train:
# #         emb_k_m = torch.tensor(emb_m[k], dtype=torch.float32).unsqueeze(0)
# #         emb_k_s = torch.tensor(emb_s[k], dtype=torch.float32).unsqueeze(0)
# #         emb_v_m = torch.tensor(emb_m[v], dtype=torch.float32).unsqueeze(0)
# #         emb_v_s = torch.tensor(emb_s[v], dtype=torch.float32).unsqueeze(0)
# #
# #         emb_k = moe_model(emb_k_m, emb_k_s).detach().numpy()
# #         emb_v = moe_model(emb_v_m, emb_v_s).detach().numpy()
# #
# #         emb_k = emb_k.squeeze()
# #         emb_v = emb_v.squeeze()
# #
# #         traindata.append([emb_k, emb_v])
# #     traindata = np.array(traindata)
# #
# #     testdata = []
# #     for k, v in test:
# #         emb_k_m = torch.tensor(emb_m[k], dtype=torch.float32).unsqueeze(0)
# #         emb_k_s = torch.tensor(emb_s[k], dtype=torch.float32).unsqueeze(0)
# #         emb_v_m = torch.tensor(emb_m[v], dtype=torch.float32).unsqueeze(0)
# #         emb_v_s = torch.tensor(emb_s[v], dtype=torch.float32).unsqueeze(0)
# #
# #         emb_k = moe_model(emb_k_m, emb_k_s).detach().numpy()
# #         emb_v = moe_model(emb_v_m, emb_v_s).detach().numpy()
# #
# #         emb_k = emb_k.squeeze()
# #         emb_v = emb_v.squeeze()
# #
# #         testdata.append([emb_k, emb_v])
# #     testdata = np.array(testdata)
# #
# #     zx, zy = align_cca(traindata, testdata, K=K, reg=reg)
# #
# #     sim_matrix = get_sim(zx, zy, top_k=10)
# #     score = []
# #     for top_k in [1, 5, 10]:
# #         score_ = hit_precision(sim_matrix, top_k=top_k)
# #         score.append(score_)
# #
# #     mrr = compute_mrr(sim_matrix)
# #     score.append(mrr)
# #     return score
#
#
# anchors = dict(json.load(open('../data/dblp/anchors.txt', 'r')))
# print(time.ctime(), '\t # of Anchors:', len(anchors))
# g1, g2 = pickle.load(open('../data/dblp/networks', 'rb'))
# print(time.ctime(), '\t Size of two networks:', len(g1), len(g2))
# datasets = dataset(anchors)
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
#
#
#
# if __name__ == '__main__':
#     pool = Pool(min(16, os.cpu_count() - 2))
#     result = []
#
#     for seed in range(1):
#         d = 768
#         emb_m, emb_s = pickle.load(open('../emb/emb_dblp_seed_0_dim_768', 'rb'))
#         emb_m = pickle.load(open('../emb/emb_dblp1_dim', 'rb'))
#         print(emb_m)
#         print(f"emb_m shape: {emb_m.shape}")
#         print(f"emb_s shape: {emb_s.shape}")
#         emb_all = np.concatenate((emb_m, emb_s), axis=-1)
#
#         # #生成目标对
#         # attribute_pairs, structure_pairs, labels = generate_target(emb_m, emb_s, anchors)
#         #
#         # # 初始化使用GlobalGatingNetwork的模型
#         # model = MixtureOfExpertsWithGlobalGating(emb_m_dim=768, emb_s_dim=768)
#         # print(f"Model initialized: {type(model)}")
#         #
#         # # 训练模型
#         # train_model(attribute_pairs, structure_pairs, labels,  model, num_epochs=10, learning_rate=0.001)
#
#         for model_idx in [2]:
#             emb = [emb_m, emb_s, emb_all][model_idx]
#             model_name = ['MAUIL-a', 'MAUIL-s', 'MAUIL'][model_idx]
#             dim = emb.shape[-1]
#             for K in [[120], [120], [120]][model_idx]:
#                 for reg in [100, 1000]:
#                     score = []
#                     seed_ = list(range(10))
#                     # psearch_partial = partial(psearch, emb_m, emb_s, K, reg, trained_model=model)
#                     # score_10 = pool.map(psearch_partial, seed_)
#                     score_10 = pool.map(partial(psearch, emb, K, reg), seed_)
#                     score_10 = np.array(score_10)
#                     assert score_10.shape == (10, 4)
#                     score = np.mean(score_10, axis=0)
#
#                     record = [seed, d, model_name, K, reg] + score.tolist()
#                     result.append(record)
#                     print(record)
#
#     json.dump(result, open('result_MAUIL_dblp.txt', 'w'))


import numpy as np
import json, pickle, time, os
from multiprocessing import Pool
from functools import partial
import torch
import torch.nn as nn
import torch.optim as optim
from align import align_cca
from utils import dataset, get_sim, hit_precision


# 定义专家网络（FFN）
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


# 定义门控网络（Router）
class Router(nn.Module):
    def __init__(self, input_dim, num_experts):
        super(Router, self).__init__()
        self.fc = nn.Linear(input_dim, num_experts)

    def forward(self, x):
        # 输出为每个专家的权重
        weights = torch.softmax(self.fc(x), dim=-1)
        return weights


class MoELayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_experts):
        super(MoELayer, self).__init__()
        self.experts = nn.ModuleList([FFN(input_dim, hidden_dim, output_dim) for _ in range(num_experts)])
        self.router = Router(input_dim, num_experts)

    def forward(self, x):
        # 计算每个专家的权重得分
        z = self.router(x)
        # 通过softmax将得分归一化为选择权重
        weights = torch.softmax(z, dim=-1)
        # print(f"Normalized weights: {weights.shape}")
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=-1)
        # print(f"Expert outputs: {expert_outputs.shape}")
        output = torch.sum(expert_outputs * weights.unsqueeze(1), dim=-1)
        # print(f"Final weighted output: {output.shape}")

        return output


class TransformerWithMoE(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_experts, num_layers):
        super(TransformerWithMoE, self).__init__()
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.MultiheadAttention(embed_dim=input_dim, num_heads=8),
                nn.LayerNorm(input_dim),
                MoELayer(input_dim * 2, hidden_dim, output_dim, num_experts),  # 注意 input_dim * 2
                nn.LayerNorm(output_dim)
            )
            for _ in range(num_layers)
        ])

    def forward(self, x_m, x_s):
        for layer in self.layers:
            # 只对其中一个嵌入使用注意力机制，保持 768 维度
            attn_output, _ = layer[0](x_m, x_m, x_m)
            x = layer[1](attn_output + x_m)  # Add & Norm

            # 将注意力机制的输出与第二个嵌入合并后输入到 MoE 层
            combined_input = torch.cat([x, x_s], dim=-1)
            moe_output = layer[2](combined_input)
            x = layer[3](moe_output + x)  # Add & Norm
        return x


def generate_target(emb_m, emb_s, anchors, num_negative_samples=1, save_path=None):
    attribute_pairs = []
    structure_pairs = []
    labels = []

    all_nodes = list(range(len(emb_m)))

    for k, v in anchors.items():
        emb_k_m = emb_m[k]
        emb_v_m = emb_m[v]
        emb_k_s = emb_s[k]
        emb_v_s = emb_s[v]

        attribute_pairs.append([emb_k_m, emb_v_m])
        structure_pairs.append([emb_k_s, emb_v_s])
        labels.append(1)

        for _ in range(num_negative_samples):
            neg_v = np.random.choice(all_nodes)
            while neg_v in anchors.values():
                neg_v = np.random.choice(all_nodes)

            neg_emb_v_m = emb_m[neg_v]
            neg_emb_v_s = emb_s[neg_v]

            attribute_pairs.append([emb_k_m, neg_emb_v_m])
            structure_pairs.append([emb_k_s, neg_emb_v_s])
            labels.append(0)

    attribute_pairs = np.array(attribute_pairs)
    structure_pairs = np.array(structure_pairs)
    labels = np.array(labels)

    # 如果提供了保存路径，则保存生成的目标对
    if save_path:
        with open(save_path, 'wb') as f:
            pickle.dump((attribute_pairs, structure_pairs, labels), f)
        print(f"Data saved to {save_path}")

    return attribute_pairs, structure_pairs, labels


def print_expert_weights(router):
    """
    打印专家的选择权重
    :param router: MoE 模型中的 Router 模块
    """
    with torch.no_grad():
        # 获取 router 的权重
        expert_weights = router.fc.weight.detach().cpu().numpy()
        print("专家权重：")
        print(expert_weights)
        print("平均选择比率：")
        print(np.mean(expert_weights, axis=0))


def train_model(attribute_pairs, structure_pairs, labels, model, num_epochs=10, learning_rate=0.001, batch_size=32, device='cpu'):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)

    clip_value = 1.0

    model.to(device)

    print("训练前的专家权重：")
    print_expert_weights(model.layers[0][2].router)

    initial_router_weights_sum = torch.sum(model.layers[0][2].router.fc.weight).item()
    print(f"训练前的Router权重总和：{initial_router_weights_sum}")

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        for i in range(0, len(attribute_pairs), batch_size):
            batch_attr_pairs = attribute_pairs[i:i+batch_size]
            batch_struct_pairs = structure_pairs[i:i+batch_size]
            batch_labels = labels[i:i+batch_size]

            batch_attr_pairs = torch.tensor(batch_attr_pairs, dtype=torch.float32).to(device)
            batch_struct_pairs = torch.tensor(batch_struct_pairs, dtype=torch.float32).to(device)
            batch_labels = torch.tensor(batch_labels, dtype=torch.float32).to(device)

            emb_k_m = batch_attr_pairs[:, 0, :]
            emb_v_m = batch_attr_pairs[:, 1, :]
            emb_k_s = batch_struct_pairs[:, 0, :]
            emb_v_s = batch_struct_pairs[:, 1, :]

            output_k = model(emb_k_m, emb_k_s)
            output_v = model(emb_v_m, emb_v_s)


            logits = torch.cat([output_k, output_v], dim=-1)

            logits = torch.sum(logits, dim=-1)

            loss = criterion(logits, batch_labels)

            optimizer.zero_grad()
            loss.backward()

            # 应用梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)

            total_loss += loss.item()
            optimizer.step()

        # 更新学习率调度器
        scheduler.step(total_loss / len(attribute_pairs))

        # 打印每个epoch结束时的专家权重
        print(f"Epoch [{epoch+1}/{num_epochs}] 结束后的专家权重：")
        print_expert_weights(model.layers[0][2].router)
        print(f"训练后的Router权重总和：{torch.sum(model.layers[0][2].router.fc.weight).item()}")
        # 打印每个epoch的平均损失
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(attribute_pairs):.4f}')

    # 打印最终的全局权重
    final_weights = model.layers[0][2].router.fc.weight.detach().cpu().numpy()
    print(f"Final global weights shape after training: {final_weights.shape}")



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


def psearch(emb_m, emb_s, K, reg, seed, trained_model):
    moe_model = trained_model
    moe_model.eval()

    test = datasets.get('test', n=2000, seed=seed)
    train = datasets.get('train', n=850, seed=seed)
    print(f"train is {train}")
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
    print(f"traindata is {traindata}")
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

if __name__ == '__main__':
    pool = Pool(min(16, os.cpu_count() - 2))
    result = []

    for seed in range(1):
        d = 768  # 输入维度
        emb_m, emb_s = pickle.load(open('../emb/emb_dblp_seed_0_dim_768', 'rb'))
        emb_m = pickle.load(open('../emb/emb_dblp1_dim', 'rb'))
        print(f"emb_m shape: {emb_m.shape}")
        print(f"emb_s shape: {emb_s.shape}")
        emb_all = np.concatenate((emb_m, emb_s), axis=-1)

        generated_targets_save_path = 'generated_targets.pkl'

        if os.path.exists(generated_targets_save_path):
            with open(generated_targets_save_path, 'rb') as f:
                attribute_pairs, structure_pairs, labels = pickle.load(f)
            print(f"Data loaded from {generated_targets_save_path}")
        else:
            attribute_pairs, structure_pairs, labels = generate_target(emb_m, emb_s, anchors, num_negative_samples=2800,
                                                                       save_path=generated_targets_save_path)
        print(attribute_pairs.shape)
        print(labels)
        model = TransformerWithMoE(input_dim=768, hidden_dim=512, output_dim=768, num_experts=4, num_layers=8)
        print(f"Model initialized: {type(model)}")

        # 训练模型
        train_model(attribute_pairs, structure_pairs, labels, model, num_epochs=2, learning_rate=0.001)

        # 对模型进行测试和评估
        for model_idx in [0]:
            emb = [emb_m, emb_s, emb_all][model_idx]
            model_name = ['MAUIL-a', 'MAUIL-s', 'MAUIL'][model_idx]
            dim = emb.shape[-1]
            for K in [[120], [120], [120]][model_idx]:
                for reg in [1000]:
                    score = []
                    seed_ = list(range(10))
                    psearch_partial = partial(psearch, emb_m, emb_s, K, reg, trained_model=model)
                    score_10 = pool.map(psearch_partial, seed_)
                    score_10 = np.array(score_10)
                    assert score_10.shape == (10, 4)
                    score = np.mean(score_10, axis=0)

                    record = [seed, d, model_name, K, reg] + score.tolist()
                    result.append(record)
                    print(record)

    # 将结果保存到文件中
    json.dump(result, open('result_MAUIL_dblp.txt', 'w'))
