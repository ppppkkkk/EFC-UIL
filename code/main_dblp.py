'''
import numpy as np
import json, pickle, time, os
from align import align_cca
from utils import dataset, get_sim, hit_precision

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

def psearch(n_train, emb, K, reg, seed=42):
    test = datasets.get('test', n=70, seed=seed)
    train = datasets.get('train', n=n_train, seed=seed)

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


anchors = dict(json.load(open('../data/dblp/dblp_1/anchors.txt', 'r')))
print(time.ctime(), '\t # of Anchors:', len(anchors))
g1, g2 = pickle.load(open('../data/dblp/dblp_1/networks', 'rb'))
print(time.ctime(), '\t Size of two networks:', len(g1), len(g2))
datasets = dataset(anchors)

if __name__ == '__main__':
    result = []
    emb_g1 = pickle.load(open('initial_embeddings1_dblp_1.pkl', 'rb'))
    emb_g2 = pickle.load(open('initial_embeddings2_dblp_1.pkl','rb'))
    emb_attr = pickle.load(open('mauil_a_dblp_1.pkl', 'rb'))
    emb_g1.update(emb_g2)
    emb_s = np.array([emb_g1[str(i)] for i in range(len(emb_g1))])

    emb_attr = (emb_attr - np.mean(emb_attr, axis=0, keepdims=True)) / np.std(emb_attr, axis=0, keepdims=True)
    emb_s = (emb_s - np.mean(emb_s, axis=0, keepdims=True)) / np.std(emb_s, axis=0, keepdims=True)
    for seed in range(3):
        d = 768

        emb_all = np.concatenate((emb_attr, emb_s), axis=-1)
        for model in [2]:
            n_train = 630
            emb = [emb_attr, emb_s, emb_all][model]
            model_name = ['MAUIL-a', 'MAUIL-s', 'MAUIL'][model]
            dim = emb.shape[-1]
            for K in [[0], [120], [120]][model]:
                for reg in [1000, 1000]:
                    score = []
                    score_result = psearch(n_train, emb, K, reg)

                    score = np.array(score_result)
                    assert score.shape == (4,)

                    score = np.round(score, 4)

                    record = [42, dim, model_name, K, reg] + score.tolist()
                    result.append(record)
                    print(record)

    json.dump(result, open('result_MAUIL_dblp.txt', 'w'))





'''

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
import networkx as nx


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


class MoESequentialLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_experts, num_layers):
        super(MoESequentialLayer, self).__init__()
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

            x_m = layer[0](x_m)
            x_s = layer[0](x_s)

            combined_input = torch.cat([x_m, x_s], dim=-1)
            moe_output = layer[1](x_m, x_s)

            final_output = layer[2](combined_input + moe_output)

        return final_output


def train_model(attribute_pairs, structure_pairs, model, num_epochs=10, learning_rate=0.001, batch_size=32, device='cpu'):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

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

            l2_loss = torch.norm(output_k - output_v, p=2, dim=-1)
            loss = torch.mean(l2_loss)

            optimizer.zero_grad()
            loss.backward()

            total_loss += loss.item()
            optimizer.step()

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
    # 设置设备为cuda或cpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 确保模型在设备上
    moe_model = trained_model.to(device)
    moe_model.eval()

    # 加载train和test anchors
    with open(f'train_anchors_{dataset}_{train_ratio}.pkl', 'rb') as f:
        train = pickle.load(f)
    with open(f'test_anchors_{dataset}_{train_ratio}.pkl', 'rb') as f:
        test = pickle.load(f)

    # 处理训练数据
    traindata = []
    for k, v in train.items():
        emb_k_m = torch.tensor(emb_a[k], dtype=torch.float32).unsqueeze(0).to(device)
        emb_k_s = torch.tensor(emb_s[k], dtype=torch.float32).unsqueeze(0).to(device)
        emb_v_m = torch.tensor(emb_a[v], dtype=torch.float32).unsqueeze(0).to(device)
        emb_v_s = torch.tensor(emb_s[v], dtype=torch.float32).unsqueeze(0).to(device)

        # 将模型和张量都放在同一设备上
        emb_k = moe_model(emb_k_m, emb_k_s).detach().cpu().numpy()
        emb_v = moe_model(emb_v_m, emb_v_s).detach().cpu().numpy()

        emb_k = emb_k.squeeze()
        emb_v = emb_v.squeeze()

        traindata.append([emb_k, emb_v])
    traindata = np.array(traindata)

    # 处理测试数据
    testdata = []
    for k, v in test.items():
        emb_k_m = torch.tensor(emb_a[k], dtype=torch.float32).unsqueeze(0).to(device)
        emb_k_s = torch.tensor(emb_s[k], dtype=torch.float32).unsqueeze(0).to(device)
        emb_v_m = torch.tensor(emb_a[v], dtype=torch.float32).unsqueeze(0).to(device)
        emb_v_s = torch.tensor(emb_s[v], dtype=torch.float32).unsqueeze(0).to(device)

        emb_k = moe_model(emb_k_m, emb_k_s).detach().cpu().numpy()
        emb_v = moe_model(emb_v_m, emb_v_s).detach().cpu().numpy()

        emb_k = emb_k.squeeze()
        emb_v = emb_v.squeeze()

        testdata.append([emb_k, emb_v])
    testdata = np.array(testdata)

    # CCA对齐
    zx, zy = align_cca(traindata, testdata, K=K, reg=reg)

    # 计算相似度矩阵
    sim_matrix = get_sim(zx, zy, top_k=10)

    # 计算评分
    score = []
    for top_k in [1, 5, 10]:
        score_ = hit_precision(sim_matrix, top_k=top_k)
        score.append(score_)

    mrr = compute_mrr(sim_matrix)
    score.append(mrr)

    return score


def psearch_without_moe(emb, K, reg, dataset, train_ratio, seed):
    with open(f'train_anchors_{dataset}_{train_ratio}.pkl', 'rb') as f:
        train = pickle.load(f)
    with open(f'test_anchors_{dataset}_{train_ratio}.pkl', 'rb') as f:
        test = pickle.load(f)

    traindata = []
    for k, v in train.items():  # 假设 train 是一个字典
        traindata.append([emb[k], emb[v]])
    traindata = np.array(traindata)

    testdata = []
    for k, v in test.items():  # 假设 test 是一个字典
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


class ContrastiveLearningModel(nn.Module):
    def __init__(self, initial_embeddings):
        super(ContrastiveLearningModel, self).__init__()
        self.embeddings = nn.Parameter(torch.tensor(initial_embeddings, dtype=torch.float32))

    def forward(self):
        return self.embeddings


def contrastive_loss(pos_embeddings, neg_embeddings, initial_embeddings, temperature=0.05):
    pos_similarity = nn.functional.cosine_similarity(initial_embeddings, pos_embeddings)
    neg_similarity = nn.functional.cosine_similarity(initial_embeddings.unsqueeze(1), neg_embeddings)
    neg_similarity_sum = torch.sum(torch.exp(neg_similarity / temperature), dim=1)
    pos_exp = torch.exp(pos_similarity / temperature)
    loss = -torch.log(pos_exp / (pos_exp + neg_similarity_sum))
    return loss.mean()


def EFCUIL(G1, G2, anchors, model_moe, batch_size=20, temperature=0.05, epochs=20, lr_a=0.005, lr_s=0.001, lr_moe=1e-6, train_ratio=0.7,
                emb_a_contrastive=True, emb_s_contrastive=True,
                moe=True,
                initial_embed_emb_a_path='initial_embeddings.pkl',
                initial_embed_emb_s_path1='initial_embeddings1.pkl',
                initial_embed_emb_s_path2='initial_embeddings2.pkl',
                train_anchors_path='train_anchors.pkl', test_anchors_path='test_anchors.pkl'):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # 加载文档嵌入
    if os.path.exists(initial_embed_emb_a_path):
        with open(initial_embed_emb_a_path, 'rb') as f:
            numpy_array_initial = pickle.load(f)
        print(f"Loaded initial embeddings from {initial_embed_emb_a_path}")

    # 加载网络嵌入
    if os.path.exists(initial_embed_emb_s_path1) and os.path.exists(initial_embed_emb_s_path2):
        with open(initial_embed_emb_s_path1, 'rb') as f1, open(initial_embed_emb_s_path2, 'rb') as f2:
            embeddings1 = pickle.load(f1)
            embeddings2 = pickle.load(f2)
        print(f"Loaded initial embeddings from {initial_embed_emb_s_path1} and {initial_embed_emb_s_path2}")

    # 文档嵌入部分
    embeddings_a1 = numpy_array_initial[:len(G1.nodes())]
    print(embeddings_a1.shape)
    embeddings_a2 = numpy_array_initial[len(G1.nodes()):]
    print(embeddings_a2.shape)
    embeddings1_array = np.array([v for v in embeddings1.values()])
    embeddings2_array = np.array([v for v in embeddings2.values()])
    #dblp_1用00, dblp_2用11,wd用01
    # if emb_a_contrastive:
    #     embeddings_a1 = (embeddings_a1 - np.mean(embeddings_a1, axis=0, keepdims=True)) / np.std(
    #         embeddings_a1, axis=0, keepdims=True)
    #     embeddings_a2 = (embeddings_a2 - np.mean(embeddings_a2, axis=0, keepdims=True)) / np.std(
    #         embeddings_a2, axis=0, keepdims=True)
    # if emb_s_contrastive:
    #     embeddings1_array = (embeddings1_array - np.mean(embeddings1_array, axis=0, keepdims=True)) / np.std(
    #         embeddings1_array, axis=0, keepdims=True)
    #     embeddings2_array = (embeddings2_array - np.mean(embeddings2_array, axis=0, keepdims=True)) / np.std(
    #         embeddings2_array, axis=0, keepdims=True)

    contrastive_model_a1 = ContrastiveLearningModel(embeddings_a1).to(device)
    contrastive_model_a2 = ContrastiveLearningModel(embeddings_a2).to(device)
    contrastive_embeddings1 = ContrastiveLearningModel(embeddings1_array).to(device)
    contrastive_embeddings2 = ContrastiveLearningModel(embeddings2_array).to(device)
    model_moe.to(device)

    optimizer_a1 = torch.optim.Adam(contrastive_model_a1.parameters(), lr=lr_a)
    optimizer_a2 = torch.optim.Adam(contrastive_model_a2.parameters(), lr=lr_a)
    optimizer2 = torch.optim.Adam(contrastive_embeddings1.parameters(), lr=lr_s)
    optimizer3 = torch.optim.Adam(contrastive_embeddings2.parameters(), lr=lr_s)
    optimizer_moe = torch.optim.Adam(model_moe.parameters(), lr=lr_moe)

    if os.path.exists(train_anchors_path) and os.path.exists(test_anchors_path):
        # 文件存在，直接读取
        with open(train_anchors_path, 'rb') as f:
            train_anchors = pickle.load(f)
        with open(test_anchors_path, 'rb') as f:
            test_anchors = pickle.load(f)
        print(f"Loaded train_anchors from {train_anchors_path} and test_anchors from {test_anchors_path}")
    else:
        # 文件不存在，生成并保存
        anchor_items = list(anchors.items())
        random.shuffle(anchor_items)

        num_train_anchors = int(train_ratio * len(anchor_items))  # 抽取 train_ratio 作为训练数据
        train_anchors = dict(anchor_items[:num_train_anchors])
        test_anchors = dict(anchor_items[num_train_anchors:])  # 剩余部分作为测试数据

        # 存储 train_anchors 和 test_anchors
        with open(train_anchors_path, 'wb') as f:
            pickle.dump(train_anchors, f)
        with open(test_anchors_path, 'wb') as f:
            pickle.dump(test_anchors, f)

        print(f"Generated and saved train_anchors to {train_anchors_path} and test_anchors to {test_anchors_path}")

    # 开始网络间的对比学习
    if not nx.is_directed(G1):
        G1 = G1.to_directed()
    if not nx.is_directed(G2):
        G2 = G2.to_directed()

    node_to_idx_G1 = {node: idx for idx, node in enumerate(G1.nodes())}
    node_to_idx_G2 = {node: idx for idx, node in enumerate(G2.nodes())}

    for epoch in range(epochs):
        model_moe.train()
        combined_loss = None

        optimizer_a1.zero_grad()
        optimizer_a2.zero_grad()
        optimizer2.zero_grad()
        optimizer3.zero_grad()
        optimizer_moe.zero_grad()

        # 文档对比学习
        if emb_a_contrastive:
            anchor_embeds1, anchor_embeds2, neg_embeds1, neg_embeds2 = [], [], [], []

            for anchor_node1, anchor_node2 in train_anchors.items():
                anchor_embeds1.append(contrastive_model_a1.embeddings[node_to_idx_G1[anchor_node1]])
                anchor_embeds2.append(contrastive_model_a2.embeddings[node_to_idx_G2[anchor_node2]])

                neighbors2 = list(G2.neighbors(anchor_node2))
                neg_embeds1.extend(
                    [contrastive_model_a2.embeddings[node_to_idx_G2[neighbor]] for neighbor in neighbors2])

                neighbors1 = list(G1.neighbors(anchor_node1))
                neg_embeds2.extend(
                    [contrastive_model_a1.embeddings[node_to_idx_G1[neighbor]] for neighbor in neighbors1])

            num_batches = len(anchor_embeds1) // batch_size + (1 if len(anchor_embeds1) % batch_size != 0 else 0)

            for batch_idx in range(num_batches):
                batch_start = batch_idx * batch_size
                batch_end = min((batch_idx + 1) * batch_size, len(anchor_embeds1))

                batch_anchor_embeds1 = torch.stack(anchor_embeds1[batch_start:batch_end]).float().to(device)
                batch_anchor_embeds2 = torch.stack(anchor_embeds2[batch_start:batch_end]).float().to(device)
                batch_neg_embeds1 = torch.stack(neg_embeds1[batch_start:batch_end]).float().to(device)
                batch_neg_embeds2 = torch.stack(neg_embeds2[batch_start:batch_end]).float().to(device)

                loss_m = contrastive_loss(batch_anchor_embeds1, batch_neg_embeds1, batch_anchor_embeds2,
                                          temperature=0.1) + \
                         contrastive_loss(batch_anchor_embeds2, batch_neg_embeds2, batch_anchor_embeds1,
                                          temperature=0.1)

                if combined_loss is None:  # 初始化 combined_loss
                    combined_loss = loss_m
                else:
                    combined_loss += loss_m

        # 网络对比学习
        if emb_s_contrastive:
            anchor_embeds1, anchor_embeds2, neg_embeds1, neg_embeds2 = [], [], [], []

            for anchor_node1, anchor_node2 in train_anchors.items():
                anchor_embeds1.append(contrastive_embeddings1.embeddings[node_to_idx_G1[anchor_node1]])
                anchor_embeds2.append(contrastive_embeddings2.embeddings[node_to_idx_G2[anchor_node2]])

                neighbors2 = list(G2.neighbors(anchor_node2))
                neg_embeds1.extend(
                    [contrastive_embeddings2.embeddings[node_to_idx_G2[neighbor]] for neighbor in neighbors2])

                neighbors1 = list(G1.neighbors(anchor_node1))
                neg_embeds2.extend(
                    [contrastive_embeddings1.embeddings[node_to_idx_G1[neighbor]] for neighbor in neighbors1])

            num_batches = len(anchor_embeds1) // batch_size + (1 if len(anchor_embeds1) % batch_size != 0 else 0)

            for batch_idx in range(num_batches):
                batch_start = batch_idx * batch_size
                batch_end = min((batch_idx + 1) * batch_size, len(anchor_embeds1))

                batch_anchor_embeds1 = torch.stack(anchor_embeds1[batch_start:batch_end]).float().to(device)
                batch_anchor_embeds2 = torch.stack(anchor_embeds2[batch_start:batch_end]).float().to(device)
                batch_neg_embeds1 = torch.stack(neg_embeds1[batch_start:batch_end]).float().to(device)
                batch_neg_embeds2 = torch.stack(neg_embeds2[batch_start:batch_end]).float().to(device)

                loss_s = contrastive_loss(batch_anchor_embeds1, batch_neg_embeds1, batch_anchor_embeds2,
                                          temperature=temperature) + \
                         contrastive_loss(batch_anchor_embeds2, batch_neg_embeds2, batch_anchor_embeds1,
                                          temperature=temperature)

                if combined_loss is None:  # 初始化 combined_loss
                    combined_loss = loss_s
                else:
                    combined_loss += loss_s

        if moe:
            emb_a1 = contrastive_model_a1().detach().cpu().numpy()
            emb_a2 = contrastive_model_a2().detach().cpu().numpy()
            emb_g1 = contrastive_embeddings1().detach().cpu().numpy()
            emb_g2 = contrastive_embeddings2().detach().cpu().numpy()
            emb_a = np.vstack([emb_a1, emb_a2])
            emb_s = np.vstack([emb_g1, emb_g2])
            emb_a = (emb_a - np.mean(emb_a, axis=0, keepdims=True)) / np.std(emb_a, axis=0, keepdims=True)
            emb_s = (emb_s - np.mean(emb_s, axis=0, keepdims=True)) / np.std(emb_s, axis=0, keepdims=True)
            attribute_pairs, structure_pairs = generate_pairs(train_anchors, emb_a, emb_s)

            for i in range(0, len(attribute_pairs), batch_size):
                batch_attr_pairs = attribute_pairs[i:i + batch_size]
                batch_struct_pairs = structure_pairs[i:i + batch_size]

                batch_attr_pairs = torch.tensor(batch_attr_pairs, dtype=torch.float32).to(device)
                batch_struct_pairs = torch.tensor(batch_struct_pairs, dtype=torch.float32).to(device)

                emb_k_m = batch_attr_pairs[:, 0, :]
                emb_v_m = batch_attr_pairs[:, 1, :]
                emb_k_s = batch_struct_pairs[:, 0, :]
                emb_v_s = batch_struct_pairs[:, 1, :]

                output_k = model_moe(emb_k_m, emb_k_s)
                output_v = model_moe(emb_v_m, emb_v_s)

                l2_loss = torch.norm(output_k - output_v, p=2, dim=-1)
                loss_moe = torch.mean(l2_loss)

                if combined_loss is None:  # 初始化 combined_loss
                    combined_loss = loss_moe
                else:
                    combined_loss += loss_moe

        if combined_loss is not None:
            combined_loss.backward()
            optimizer_a1.step()
            optimizer_a2.step()
            optimizer2.step()
            optimizer3.step()
            optimizer_moe.step()

            print(f"Epoch {epoch + 1}/{epochs}, Combined Loss: {combined_loss.item() / len(anchors)}")
        else:
            print(f"Epoch {epoch + 1}/{epochs}, No Loss Computed.")

    return model_moe, contrastive_model_a1().detach().cpu().numpy(), contrastive_model_a2().detach().cpu().numpy(), \
           contrastive_embeddings1().detach().cpu().numpy(), contrastive_embeddings2().detach().cpu().numpy()


anchors = dict(json.load(open(f'../data/dblp/dblp_2/anchors.txt', 'r')))
print(time.ctime(), '\t # of Anchors:', len(anchors))
g1, g2 = pickle.load(open(f'../data/dblp/dblp_2/networks', 'rb'))
print(time.ctime(), '\t Size of two networks:', len(g1), len(g2))
datasets = dataset(anchors)

if __name__ == '__main__':
    seed_value = 42
    set_seed(seed_value)
    train_ratio = 0.7
    result = []
    dataset = 'dblp_2'
    emb_a_contrastive = True
    emb_s_contrastive = True
    moe = True
    for seed in [seed_value]:
        d = 384
        model_moe = MoESequentialLayer(input_dim=d, hidden_dim=512, output_dim=d, num_experts=2, num_layers=1)
        model_moe, emb_a1, emb_a2, emb_g1, emb_g2 = EFCUIL(G1=g1, G2=g2, anchors=anchors, model_moe=model_moe,
                                                           batch_size=20, temperature=0.05, epochs=20, lr_a=0.001,
                                                           lr_s=0.00005,
                                                           lr_moe=1e-6, train_ratio=train_ratio,
                                                           emb_a_contrastive=emb_a_contrastive,
                                                           emb_s_contrastive=emb_s_contrastive,
                                                           moe=moe,
                                                           initial_embed_emb_a_path=f'word2vec_embeddings_{dataset}_{d}.pkl',
                                                           initial_embed_emb_s_path1=f'initial_embeddings1_{dataset}_{d}.pkl',
                                                           initial_embed_emb_s_path2=f'initial_embeddings2_{dataset}_{d}.pkl',
                                                           train_anchors_path=f'train_anchors_{dataset}_{train_ratio}.pkl',
                                                           test_anchors_path=f'test_anchors_{dataset}_{train_ratio}.pkl')

        emb_a = np.vstack([emb_a1, emb_a2])
        emb_s = np.vstack([emb_g1, emb_g2])

        emb_a = (emb_a - np.mean(emb_a, axis=0, keepdims=True)) / np.std(emb_a, axis=0, keepdims=True)
        emb_s = (emb_s - np.mean(emb_s, axis=0, keepdims=True)) / np.std(emb_s, axis=0, keepdims=True)

        emb_all = np.concatenate((emb_a, emb_s), axis=-1)

        for model_idx in [0]:
            model_name = ['EFC-UIL'][model_idx]
            emb = [emb_all][model_idx]
            dim = d
            for K in [[120]][model_idx]:
                for reg in [1000]:
                    score = []
                    if moe == True:
                        score_result = psearch_with_seed(emb_a=emb_a, emb_s=emb_s, K=K, reg=reg, dataset=dataset,
                                                         trained_model=model_moe, train_ratio=train_ratio,
                                                         seed=seed_value)
                    else:
                        score_result = psearch_without_moe(emb, K, reg, dataset, train_ratio, seed_value)

                    score = np.array(score_result)
                    assert score.shape == (4,)

                    score = np.round(score, 4)

                    record = [seed_value, dim, model_name, K, reg] + score.tolist()
                    result.append(record)
                    print(record)

    # 将结果保存到文件中，保留4位小数
    json.dump(result, open(f'result_EFC-UIL_{dataset}_{train_ratio}.txt', 'w'), indent=4)

