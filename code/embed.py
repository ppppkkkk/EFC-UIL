import json
from random import random
from ge import LINE
import torch
from sentence_transformers.ConSERT import ConSERT
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
import time
import numpy as np
import networkx as nx
import pickle, os
import torch.nn as nn
# stop_words = pickle.load(open('../data/wd/stop_words_cn.pkl', 'rb'))
import tensorflow as tf
tf.compat.v1.disable_v2_behavior()


# 2-layer bert encoder
class ContrastiveLearningModel(nn.Module):
    def __init__(self, initial_embeddings):
        super(ContrastiveLearningModel, self).__init__()
        # 用初始嵌入矩阵初始化模型的权重
        self.embeddings = nn.Parameter(torch.tensor(initial_embeddings, dtype=torch.float32))

    def forward(self):
        # 直接返回参数化的嵌入
        return self.embeddings


def contrastive_loss(pos_embeddings, neg_embeddings, initial_embeddings, temperature=0.1):
    # 计算初始嵌入与正样本之间的距离（最小化距离）
    pos_dist = F.cosine_similarity(initial_embeddings, pos_embeddings)
    pos_loss = -torch.mean(pos_dist)

    # 计算初始嵌入与负样本之间的距离（最大化距离）
    neg_dist = F.cosine_similarity(initial_embeddings, neg_embeddings)
    neg_loss = torch.mean(neg_dist)

    # 总损失是正样本损失和负样本损失之和
    loss = pos_loss + neg_loss
    return loss


def batch_tokenize(model, docs, device):
    tokenized = [model.tokenize([doc]) for doc in docs]
    tokenized = [{key: value.to(device) for key, value in doc.items()} for doc in tokenized]
    return tokenized


def load_pretrained_model(model_path='consert_model_initial.pth', device=''):
    model_name = '../bert-base-uncased'

    torch.backends.cudnn.benchmark = True  # 启用 CuDNN 的自动优化
    # 初始化模型
    model = ConSERT(model_name, device=device, cutoff_rate=0.12, close_dropout=True)
    model.__setattr__("max_seq_length", 512)
    # 加载保存的初始状态
    model.load_state_dict(torch.load(model_path))

    print(f"Model loaded from: {model_path}")
    return model


def my_embed(docs, batch_size=20, temperature=0.1, epochs=20, learning_rate=0.0001,
             contrastive_model_path='contrastive_model_path.pth', initial_embed_path='initial_embeddings.pkl'):
    model_name = '../bert-base-uncased'
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True  # 启用 CuDNN 的自动优化

    model = ConSERT(model_name, device=device, cutoff_rate=0.12, close_dropout=True)
    model.__setattr__("max_seq_length", 512)
    model.to(device)
    # 检查是否存在初始嵌入文件
    if os.path.exists(initial_embed_path):
        with open(initial_embed_path, 'rb') as f:
            numpy_array_initial = pickle.load(f)
        print(f"Loaded initial embeddings from {initial_embed_path}")
    else:
        # 生成初始嵌入
        embed = []
        model.eval()
        with torch.no_grad():
            for i in range(0, len(docs), batch_size):
                batch_docs = docs[i:i + batch_size]
                tokenized = batch_tokenize(model, batch_docs, device)
                with autocast():
                    out_features = [model.forward(features) for features in tokenized]
                    embeddings = torch.cat([out['sentence_embedding'] for out in out_features])
                    embed.append(embeddings)
                    del embeddings
                    torch.cuda.empty_cache()

        numpy_array_initial = torch.cat(embed, dim=0).cpu().detach().numpy()

        # 保存初始嵌入
        with open(initial_embed_path, 'wb') as f:
            pickle.dump(numpy_array_initial, f)
        print(f"Initial embeddings saved to {initial_embed_path}")

    contrastive_model = ContrastiveLearningModel(numpy_array_initial).to(device)
    if os.path.exists(contrastive_model_path):
        contrastive_model.load_state_dict(torch.load(contrastive_model_path))
    optimizer = torch.optim.Adam(contrastive_model.parameters(), lr=learning_rate)
    scaler = GradScaler()

    print(time.ctime(), '\tStarting contrastive learning training...')
    contrastive_model.train()

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1} starts")
        epoch_loss = 0.0
        optimizer.zero_grad()

        for i in range(0, len(docs), batch_size):
            print(i)
            batch_docs = docs[i:i + batch_size]
            tokenized = batch_tokenize(model, batch_docs, device)

            with autocast():
                pos_out_features = [model.forward(features) for features in tokenized]
                pos_embeddings = torch.cat([out['sentence_embedding'] for out in pos_out_features]).to(device)

            negative_features = []
            for doc in tokenized:
                augmented = model.data_augment(doc)
                augmented_on_device = {key: value.to(device) if isinstance(value, torch.Tensor) else value for
                                       key, value in augmented.items()}
                negative_features.append(augmented_on_device)
                del augmented_on_device
            del tokenized

            for features in negative_features:
                for key, value in features.items():
                    features[key] = value[:, :512]

            with autocast():
                neg_out_features = [model.forward(features) for features in negative_features]
                neg_embeddings = torch.cat([out['sentence_embedding'] for out in neg_out_features]).to(device)
                current_embeddings = contrastive_model()

                loss = contrastive_loss(pos_embeddings, neg_embeddings, current_embeddings[i:i + batch_size],
                                        temperature)

                print(f"Total Loss: {loss.item()}")
                epoch_loss += loss.item()

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            torch.cuda.synchronize()

            del pos_embeddings, neg_embeddings
            torch.cuda.empty_cache()
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(docs)}')

        torch.save(contrastive_model.state_dict(), contrastive_model_path)
        print(f"Model state saved")
    print("Training complete")

    numpy_array = []
    contrastive_model.eval()
    with torch.no_grad():
        for i in range(0, len(docs), batch_size):
            # 获取当前批次的嵌入
            batch_embeddings = contrastive_model()[i:i + batch_size].cpu().detach().numpy()
            print(batch_embeddings.shape)
            numpy_array.append(batch_embeddings)
            del batch_embeddings
            torch.cuda.empty_cache()

    numpy_array = np.concatenate(numpy_array, axis=0)
    print(numpy_array)
    numpy_array = (numpy_array - np.mean(numpy_array, axis=0, keepdims=True)) / np.std(numpy_array, axis=0, keepdims=True)
    pickle.dump(numpy_array, open('../emb/emb_dblp1_dim', 'wb'))
    return numpy_array


def cosine_similarity(vec1, vec2):
    return nn.functional.cosine_similarity(vec1.unsqueeze(0), vec2.unsqueeze(0)).item()


def contrastive_loss_network(anchor_embed, positive_embed, negative_embed, temperature=0.05):
    pos_similarity = nn.functional.cosine_similarity(anchor_embed, positive_embed)
    neg_similarity = nn.functional.cosine_similarity(anchor_embed, negative_embed)

    loss = -torch.log(torch.exp(pos_similarity / temperature) / (
                torch.exp(pos_similarity / temperature) + torch.exp(neg_similarity / temperature)))
    return loss.mean()


def network_embed(G1, G2, anchors, dim=768, method="line", order='all', contrastive=False, epochs=50,
                  initial_embed_path1='initial_embeddings1.pkl', initial_embed_path2='initial_embeddings2.pkl'):
    if not nx.is_directed(G1):
        G1 = G1.to_directed()
    if not nx.is_directed(G2):
        G2 = G2.to_directed()

    G1 = nx.relabel_nodes(G1, lambda x: str(x))
    G2 = nx.relabel_nodes(G2, lambda x: str(x))

    # 如果已存在初始嵌入文件，则加载它们
    if os.path.exists(initial_embed_path1) and os.path.exists(initial_embed_path2):
        with open(initial_embed_path1, 'rb') as f1, open(initial_embed_path2, 'rb') as f2:
            embeddings1 = pickle.load(f1)
            embeddings2 = pickle.load(f2)
        print(f"Loaded initial embeddings from {initial_embed_path1} and {initial_embed_path2}")
    else:
        # 生成初始嵌入
        if method == 'line':
            model1 = LINE(G1, embedding_size=dim, order=order)
            model2 = LINE(G2, embedding_size=dim, order=order)

            model1.train(batch_size=1024, epochs=50, verbose=2)
            model2.train(batch_size=1024, epochs=50, verbose=2)

            embeddings1 = model1.get_embeddings()
            embeddings2 = model2.get_embeddings()

            # 保存初始嵌入
            with open(initial_embed_path1, 'wb') as f1, open(initial_embed_path2, 'wb') as f2:
                pickle.dump(embeddings1, f1)
                pickle.dump(embeddings2, f2)
            print(f"Initial embeddings saved to {initial_embed_path1} and {initial_embed_path2}")

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # 对比学习处理
    if contrastive:
        for epoch in range(epochs):
            total_loss = 0.0

            anchor_embeds1, anchor_embeds2, neg_embeds1, neg_embeds2 = [], [], [], []

            for anchor_node1, anchor_node2 in anchors.items():
                anchor_node1, anchor_node2 = str(anchor_node1), str(anchor_node2)

                # 添加锚点嵌入
                anchor_embeds1.append(embeddings1[anchor_node1])
                anchor_embeds2.append(embeddings2[anchor_node2])

                # 选择与锚点最相似的非邻居节点作为负样本
                neighbors1 = list(G1.neighbors(anchor_node1))
                non_neighbors1 = list(set(G1.nodes()) - set(neighbors1) - {anchor_node1})
                if non_neighbors1:
                    neg_node1 = min(non_neighbors1,
                                    key=lambda n: cosine_similarity(torch.tensor(embeddings1[str(n)]).float(),
                                                                    torch.tensor(embeddings1[anchor_node1]).float()))
                    neg_embeds1.append(embeddings1[str(neg_node1)])

                neighbors2 = list(G2.neighbors(anchor_node2))
                non_neighbors2 = list(set(G2.nodes()) - set(neighbors2) - {anchor_node2})
                if non_neighbors2:
                    neg_node2 = min(non_neighbors2,
                                    key=lambda n: cosine_similarity(torch.tensor(embeddings2[str(n)]).float(),
                                                                    torch.tensor(embeddings2[anchor_node2]).float()))
                    neg_embeds2.append(embeddings2[str(neg_node2)])

            # 将嵌入转换为 torch tensor 并在批处理上计算损失
            anchor_embeds1 = torch.tensor(anchor_embeds1).float().to(device)
            anchor_embeds2 = torch.tensor(anchor_embeds2).float().to(device)
            neg_embeds1 = torch.tensor(neg_embeds1).float().to(device)
            neg_embeds2 = torch.tensor(neg_embeds2).float().to(device)

            loss = contrastive_loss_network(anchor_embeds1, anchor_embeds2, neg_embeds1, temperature=0.05) + \
                   contrastive_loss_network(anchor_embeds2, anchor_embeds1, neg_embeds2, temperature=0.05)

            total_loss += loss.item()

            # 批量处理后更新嵌入
            with torch.no_grad():
                for i, anchor_node1 in enumerate(anchors.keys()):
                    embeddings1[str(anchor_node1)] -= loss.item() * 0.001
                    embeddings2[str(anchors[anchor_node1])] -= loss.item() * 0.001

            print(f'Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(anchors)}')

    return embeddings1, embeddings2


def embed_dblp():
    print(time.ctime(), '\tLoading data...')
    g1, g2 = pickle.load(open('../data/dblp/networks', 'rb'))
    print(time.ctime(), '\t Size of two networks:', len(g1), len(g2))
    attrs = pickle.load(open('../data/dblp/attrs', 'rb'))
    topic = []
    print(len(attrs))

    for i in range(len(attrs)):
        v = attrs[i]
        topic.append(v[0])

    for seed in range(1):
        for d in [768]:
            print(time.ctime(), '\tMy level attributes embedding...')

            anchors = dict(json.load(open('../data/dblp/anchors.txt', 'r')))
            print(anchors)
            print(time.ctime(), '\t # of Anchors:', len(anchors))
            print(time.ctime(), '\tNetwork embedding...')
            emb_g1, emb_g2 = network_embed(g1, g2, anchors, dim=768,
                                           method="line", order='all', contrastive=True, epochs=50)
            emb_g1.update(emb_g2)
            emb_s = np.array([emb_g1[str(i)] for i in range(len(emb_g1))])

            emb_s = (emb_s - np.mean(emb_s, axis=0, keepdims=True)) / np.std(emb_s, axis=0, keepdims=True)
            #emb_m = (emb_m - np.mean(emb_m, axis=0, keepdims=True) / np.std(emb_m, axis=0, keepdims=True))

            # Saving embeddings
            pickle.dump(emb_s, open('../emb/emb_dblp_seed_{}_dim_{}'.format(seed, d), 'wb'))
            #pickle.dump((emb_m, emb_s), open('../emb/emb_dblp_seed_{}_dim_{}'.format(seed, d), 'wb'))




os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
inputs = 0
if int(inputs) == 0:
    print('Embedding dataset: dblp')
    embed_dblp()
else:
    print('Embedding dataset: wd')
