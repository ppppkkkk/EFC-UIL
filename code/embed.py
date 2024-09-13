import tensorflow as tf
from ge import LINE
import torch
from sentence_transformers.ConSERT import ConSERT
from torch.cuda.amp import autocast
import networkx as nx
import os
import torch.nn as nn
import copy
import random
import time
import pickle
import json
import numpy as np
import jieba
import zhconv
import re
from gensim.models import Word2Vec

stop_words = pickle.load(open('../data/wd/stop_words_cn.pkl', 'rb'))
tf.compat.v1.disable_v2_behavior()


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


def batch_tokenize(model, docs, device):
    tokenized = [model.tokenize([doc]) for doc in docs]
    tokenized = [{key: value.to(device) for key, value in doc.items()} for doc in tokenized]
    return tokenized


def word2vec_embed(docs, embed_size=768, window=5, min_count=1, workers=4,
                   initial_embed_path='.pkl'):
    # 先检查是否存在已经保存的嵌入
    if os.path.exists(initial_embed_path):
        with open(initial_embed_path, 'rb') as f:
            embeddings = pickle.load(f)
        print(f"Loaded word2vec embeddings from {initial_embed_path}")
        return embeddings

    # 分词
    tokenized_docs = [doc.split() for doc in docs]

    # 训练 Word2Vec 模型
    print("Training Word2Vec model...")
    word2vec_model = Word2Vec(sentences=tokenized_docs, vector_size=embed_size, window=window, min_count=min_count,
                              workers=workers)

    # 为每个文档生成嵌入
    print("Generating document embeddings...")
    embeddings = []
    for doc in tokenized_docs:
        # 过滤掉不在词汇表中的单词
        valid_words = [word for word in doc if word in word2vec_model.wv]

        if len(valid_words) > 0:
            # 计算文档中每个单词的嵌入向量的平均值作为文档的嵌入
            doc_embedding = np.mean([word2vec_model.wv[word] for word in valid_words], axis=0)
            embeddings.append(doc_embedding)
        else:
            # 如果文档为空或没有有效的词汇，使用零向量填充
            embeddings.append(np.zeros(embed_size))

    embeddings = np.array(embeddings)

    # 保存嵌入到文件
    with open(initial_embed_path, 'wb') as f:
        pickle.dump(embeddings, f)
    print(f"Word2Vec embeddings saved to {initial_embed_path}")

    return embeddings


def my_embed(docs, G1, G2, anchors, batch_size=20, temperature=0.05, epochs=20, learning_rate=0.001,
             between_network_contrastive=True, initial_embed_path='initial_embeddings_dblp_1.pkl', type='eng'):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if type == 'eng':
        model_name = '../bert-base-uncased'
    else:
        model_name = '../chinese_wwm_ext'
    torch.backends.cudnn.benchmark = True  # 启用 CuDNN 的自动优化

    model = ConSERT(model_name, device=device)
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

        with open(initial_embed_path, 'wb') as f:
            pickle.dump(numpy_array_initial, f)
        print(f"Initial embeddings saved to {initial_embed_path}")

    # 生成 contrastive_model
    contrastive_model = ContrastiveLearningModel(numpy_array_initial).to(device)

    optimizer = torch.optim.Adam(contrastive_model.parameters(), lr=learning_rate)

    # 开始网络间的对比学习
    if not nx.is_directed(G1):
        G1 = G1.to_directed()
    if not nx.is_directed(G2):
        G2 = G2.to_directed()

    node_to_idx_G1 = {node: idx for idx, node in enumerate(G1.nodes())}
    node_to_idx_G2 = {node: idx for idx, node in enumerate(G2.nodes())}

    if between_network_contrastive:
        for epoch in range(epochs):
            total_loss = 0.0
            print(f"Epoch {epoch + 1}/{epochs}")

            anchor_embeds1, anchor_embeds2, neg_embeds1, neg_embeds2 = [], [], [], []

            # 遍历锚点并生成嵌入和负样本
            for anchor_node1, anchor_node2 in anchors.items():
                anchor_node1, anchor_node2 = anchor_node1, anchor_node2

                # 直接从 contrastive_model.embeddings 中获取锚点嵌入
                anchor_embeds1.append(contrastive_model.embeddings[node_to_idx_G1[anchor_node1]])
                anchor_embeds2.append(contrastive_model.embeddings[node_to_idx_G2[anchor_node2]])

                # 选择 anchor_node2 的所有邻居作为 anchor_node1 的负样本
                neighbors2 = list(G2.neighbors(anchor_node2))
                neg_embeds1.extend([contrastive_model.embeddings[node_to_idx_G2[neighbor]] for neighbor in neighbors2])

                # 选择 anchor_node1 的所有邻居作为 anchor_node2 的负样本
                neighbors1 = list(G1.neighbors(anchor_node1))
                neg_embeds2.extend([contrastive_model.embeddings[node_to_idx_G1[neighbor]] for neighbor in neighbors1])

            # 分批处理锚点和负样本
            num_batches = len(anchor_embeds1) // batch_size + (1 if len(anchor_embeds1) % batch_size != 0 else 0)

            for batch_idx in range(num_batches):
                batch_start = batch_idx * batch_size
                batch_end = min((batch_idx + 1) * batch_size, len(anchor_embeds1))

                # 将当前批次的嵌入转换为 numpy.ndarray，然后转换为 torch.tensor
                batch_anchor_embeds1 = torch.stack(anchor_embeds1[batch_start:batch_end]).float().to(device)
                batch_anchor_embeds2 = torch.stack(anchor_embeds2[batch_start:batch_end]).float().to(device)
                batch_neg_embeds1 = torch.stack(neg_embeds1[batch_start:batch_end]).float().to(device)
                batch_neg_embeds2 = torch.stack(neg_embeds2[batch_start:batch_end]).float().to(device)

                optimizer.zero_grad()  # 清空梯度

                # 使用自定义的对比损失函数
                loss = contrastive_loss(batch_anchor_embeds1, batch_neg_embeds1, batch_anchor_embeds2,
                                        temperature=temperature) + \
                       contrastive_loss(batch_anchor_embeds2, batch_neg_embeds2, batch_anchor_embeds1, temperature=temperature)

                total_loss += loss.item()

                # 反向传播
                loss.backward()
                optimizer.step()

            print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(anchors)}")

    print("between_network_contrastive finish")

    # 生成最终的嵌入
    final_embeddings_G1 = contrastive_model.embeddings[
        [node_to_idx_G1[node] for node in G1.nodes()]].cpu().detach().numpy()
    final_embeddings_G2 = contrastive_model.embeddings[
        [node_to_idx_G2[node] for node in G2.nodes()]].cpu().detach().numpy()

    # 将 G1 和 G2 的最终嵌入拼接在一起
    final_embeddings = np.concatenate([final_embeddings_G1, final_embeddings_G2], axis=0)

    print(f"Final concatenated embeddings shape: {final_embeddings.shape}")

    return final_embeddings


def densify_graphs(G1, G2, anchors):
    # 为 G1 和 G2 创建副本，用于读取邻居信息
    G1_copy = copy.deepcopy(G1)
    G2_copy = copy.deepcopy(G2)

    for anchor_node1, anchor_node2 in anchors.items():
        anchor_node1, anchor_node2 = str(anchor_node1), str(anchor_node2)

        # G1 副本中 A1 的邻居
        neighbors1 = list(G1_copy.neighbors(anchor_node1))
        for neighbor in neighbors1:
            if neighbor in anchors:  # 该邻居是锚节点
                corresponding_neighbor = anchors[neighbor]  # 找到对应 G2 中的锚节点

                # 如果 G2 副本中 B1 的邻居没有该对应锚节点，则在 G2 上添加一条边
                if corresponding_neighbor not in G2_copy.neighbors(anchor_node2):
                    G2.add_edge(anchor_node2, corresponding_neighbor)
                    G2.add_edge(corresponding_neighbor, anchor_node2)  # 添加双向边

        # G2 副本中 B1 的邻居
        neighbors2 = list(G2_copy.neighbors(anchor_node2))
        for neighbor in neighbors2:
            if neighbor in anchors.values():  # 该邻居是锚节点
                corresponding_neighbor = list(anchors.keys())[list(anchors.values()).index(neighbor)]  # 找到对应 G1 中的锚节点

                # 如果 G1 副本中 A1 的邻居没有该对应锚节点，则在 G1 上添加一条边
                if corresponding_neighbor not in G1_copy.neighbors(anchor_node1):
                    G1.add_edge(anchor_node1, corresponding_neighbor)
                    G1.add_edge(corresponding_neighbor, anchor_node1)  # 添加双向边

    print("G1 and G2 densification complete")
    return G1, G2


def network_embed(G1, G2, anchors, dim=768, batch_size=20,
                  contrastive=False, epochs=10, learning_rate=0.001,
                  initial_embed_path1='initial_embeddings1_dblp_1.pkl',
                  initial_embed_path2='initial_embeddings2_dblp_1.pkl'):
    if not nx.is_directed(G1):
        G1 = G1.to_directed()
    if not nx.is_directed(G2):
        G2 = G2.to_directed()

    G1 = nx.relabel_nodes(G1, lambda x: str(x))
    G2 = nx.relabel_nodes(G2, lambda x: str(x))

    # 创建节点到索引的映射
    node_to_idx_G1 = {node: idx for idx, node in enumerate(G1.nodes())}
    node_to_idx_G2 = {node: idx for idx, node in enumerate(G2.nodes())}

    # 图 densification 处理
    G1, G2 = densify_graphs(G1, G2, anchors)

    # 随机抽取 30% 锚点用于训练
    anchor_items = list(anchors.items())
    num_train_anchors = int(0.3 * len(anchor_items))  # 抽取30%
    train_anchors = dict(random.sample(anchor_items, num_train_anchors))  # 生成训练用的锚点字典

    print(f"Using {len(train_anchors)} out of {len(anchors)} anchors for training.")

    # 如果已存在初始嵌入文件，则加载它们
    if os.path.exists(initial_embed_path1) and os.path.exists(initial_embed_path2):
        with open(initial_embed_path1, 'rb') as f1, open(initial_embed_path2, 'rb') as f2:
            embeddings1 = pickle.load(f1)
            embeddings2 = pickle.load(f2)
        print(f"Loaded initial embeddings from {initial_embed_path1} and {initial_embed_path2}")
    else:
        # 生成初始嵌入
        model1 = LINE(G1, embedding_size=dim, order="all")
        model2 = LINE(G2, embedding_size=dim, order="all")

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

    # 将 embeddings1 和 embeddings2 转换为 numpy 数组并确保顺序正确
    embeddings1 = np.array([embeddings1[node] for node in G1.nodes()])
    embeddings2 = np.array([embeddings2[node] for node in G2.nodes()])

    # 使用 ContrastiveLearningModel 初始化嵌入
    model1 = ContrastiveLearningModel(embeddings1).to(device)
    model2 = ContrastiveLearningModel(embeddings2).to(device)

    optimizer1 = torch.optim.Adam(model1.parameters(), lr=learning_rate)
    optimizer2 = torch.optim.Adam(model2.parameters(), lr=learning_rate)

    # 对比学习处理
    if contrastive:
        for epoch in range(epochs):
            total_loss = 0.0
            print(f"Epoch {epoch + 1}/{epochs}")

            anchor_embeds1, anchor_embeds2, neg_embeds1, neg_embeds2 = [], [], [], []

            # 使用训练锚点进行训练
            for anchor_node1, anchor_node2 in train_anchors.items():
                anchor_node1, anchor_node2 = str(anchor_node1), str(anchor_node2)

                # 使用节点到索引的映射访问嵌入
                anchor_embeds1.append(model1.embeddings[node_to_idx_G1[anchor_node1]])
                anchor_embeds2.append(model2.embeddings[node_to_idx_G2[anchor_node2]])

                # 选择 anchor_node2 的所有邻居作为 anchor_node1 的负样本
                neighbors2 = list(G2.neighbors(anchor_node2))
                neg_embeds1.extend([model2.embeddings[node_to_idx_G2[neighbor]] for neighbor in neighbors2])

                # 选择 anchor_node1 的所有邻居作为 anchor_node2 的负样本
                neighbors1 = list(G1.neighbors(anchor_node1))
                neg_embeds2.extend([model1.embeddings[node_to_idx_G1[neighbor]] for neighbor in neighbors1])

            # 分批处理锚点和负样本
            num_batches = len(anchor_embeds1) // batch_size + (1 if len(anchor_embeds1) % batch_size != 0 else 0)

            for batch_idx in range(num_batches):
                batch_start = batch_idx * batch_size
                batch_end = min((batch_idx + 1) * batch_size, len(anchor_embeds1))

                # 将当前批次的嵌入转换为 torch.tensor
                batch_anchor_embeds1 = torch.stack(anchor_embeds1[batch_start:batch_end]).float().to(device)
                batch_anchor_embeds2 = torch.stack(anchor_embeds2[batch_start:batch_end]).float().to(device)
                batch_neg_embeds1 = torch.stack(neg_embeds1[batch_start:batch_end]).float().to(device)
                batch_neg_embeds2 = torch.stack(neg_embeds2[batch_start:batch_end]).float().to(device)

                optimizer1.zero_grad()  # 清空梯度
                optimizer2.zero_grad()  # 清空梯度

                # 使用自定义的对比损失函数，并将两个 loss 合并
                loss1 = contrastive_loss(batch_anchor_embeds1, batch_neg_embeds1, batch_anchor_embeds2,
                                         temperature=0.05)
                loss2 = contrastive_loss(batch_anchor_embeds2, batch_neg_embeds2, batch_anchor_embeds1,
                                         temperature=0.05)

                combined_loss = loss1 + loss2
                total_loss += combined_loss.item()

                # 反向传播和优化器更新
                combined_loss.backward()
                optimizer1.step()
                optimizer2.step()

            print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(train_anchors)}")

    # 获取最终的嵌入
    final_embeddings1 = model1.embeddings.detach().cpu().numpy()
    final_embeddings2 = model2.embeddings.detach().cpu().numpy()

    return final_embeddings1, final_embeddings2


def joint_embed(G1, G2, anchors, batch_size=20, temperature=0.05, epochs=20, learning_rate=0.01,
                emb_m_contrastive=True, emb_s_contrastive=True,
                initial_embed_emb_m_path='initial_embeddings_dblp_1.pkl',
                initial_embed_emb_s_path1='initial_embeddings1_dblp_1.pkl',
                initial_embed_emb_s_path2='initial_embeddings2_dblp_1.pkl'):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # 加载文档嵌入
    if os.path.exists(initial_embed_emb_m_path):
        with open(initial_embed_emb_m_path, 'rb') as f:
            numpy_array_initial = pickle.load(f)
        print(f"Loaded initial embeddings from {initial_embed_emb_m_path}")

    # 加载网络嵌入
    if os.path.exists(initial_embed_emb_s_path1) and os.path.exists(initial_embed_emb_s_path2):
        with open(initial_embed_emb_s_path1, 'rb') as f1, open(initial_embed_emb_s_path2, 'rb') as f2:
            embeddings1 = pickle.load(f1)
            embeddings2 = pickle.load(f2)
        print(f"Loaded initial embeddings from {initial_embed_emb_s_path1} and {initial_embed_emb_s_path2}")

    embeddings1_array = np.array([v for v in embeddings1.values()])
    embeddings2_array = np.array([v for v in embeddings2.values()])

    contrastive_model = ContrastiveLearningModel(numpy_array_initial).to(device)
    contrastive_embeddings1 = ContrastiveLearningModel(embeddings1_array).to(device)
    contrastive_embeddings2 = ContrastiveLearningModel(embeddings2_array).to(device)

    optimizer1 = torch.optim.Adam(contrastive_model.parameters(), lr=learning_rate)
    optimizer2 = torch.optim.Adam(contrastive_embeddings1.parameters(), lr=learning_rate)
    optimizer3 = torch.optim.Adam(contrastive_embeddings2.parameters(), lr=learning_rate)

    anchor_items = list(anchors.items())
    num_train_anchors = int(0.3 * len(anchor_items))  # 抽取30%
    train_anchors = dict(random.sample(anchor_items, num_train_anchors))
    # 开始网络间的对比学习
    if not nx.is_directed(G1):
        G1 = G1.to_directed()
    if not nx.is_directed(G2):
        G2 = G2.to_directed()

    node_to_idx_G1 = {node: idx for idx, node in enumerate(G1.nodes())}
    node_to_idx_G2 = {node: idx for idx, node in enumerate(G2.nodes())}

    for epoch in range(epochs):
        combined_loss = None

        optimizer1.zero_grad()
        optimizer2.zero_grad()
        optimizer3.zero_grad()

        # 文档对比学习
        if emb_m_contrastive:
            anchor_embeds1, anchor_embeds2, neg_embeds1, neg_embeds2 = [], [], [], []

            for anchor_node1, anchor_node2 in anchors.items():
                anchor_embeds1.append(contrastive_model.embeddings[node_to_idx_G1[anchor_node1]])
                anchor_embeds2.append(contrastive_model.embeddings[node_to_idx_G2[anchor_node2]])

                neighbors2 = list(G2.neighbors(anchor_node2))
                neg_embeds1.extend([contrastive_model.embeddings[node_to_idx_G2[neighbor]] for neighbor in neighbors2])

                neighbors1 = list(G1.neighbors(anchor_node1))
                neg_embeds2.extend([contrastive_model.embeddings[node_to_idx_G1[neighbor]] for neighbor in neighbors1])

            num_batches = len(anchor_embeds1) // batch_size + (1 if len(anchor_embeds1) % batch_size != 0 else 0)

            for batch_idx in range(num_batches):
                batch_start = batch_idx * batch_size
                batch_end = min((batch_idx + 1) * batch_size, len(anchor_embeds1))

                batch_anchor_embeds1 = torch.stack(anchor_embeds1[batch_start:batch_end]).float().to(device)
                batch_anchor_embeds2 = torch.stack(anchor_embeds2[batch_start:batch_end]).float().to(device)
                batch_neg_embeds1 = torch.stack(neg_embeds1[batch_start:batch_end]).float().to(device)
                batch_neg_embeds2 = torch.stack(neg_embeds2[batch_start:batch_end]).float().to(device)

                loss_m = contrastive_loss(batch_anchor_embeds1, batch_neg_embeds1, batch_anchor_embeds2,
                                          temperature=temperature) + \
                         contrastive_loss(batch_anchor_embeds2, batch_neg_embeds2, batch_anchor_embeds1,
                                          temperature=temperature)

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

        if combined_loss is not None:
            combined_loss.backward()
            optimizer1.step()
            optimizer2.step()
            optimizer3.step()

            print(f"Epoch {epoch + 1}/{epochs}, Combined Loss: {combined_loss.item() / len(anchors)}")
        else:
            print(f"Epoch {epoch + 1}/{epochs}, No Loss Computed.")

    if emb_m_contrastive:
        final_embeddings_G1_m = contrastive_model.embeddings[
            [node_to_idx_G1[node] for node in G1.nodes()]].cpu().detach().numpy()
        final_embeddings_G2_m = contrastive_model.embeddings[
            [node_to_idx_G2[node] for node in G2.nodes()]].cpu().detach().numpy()
        final_embeddings_m = np.concatenate([final_embeddings_G1_m, final_embeddings_G2_m], axis=0)
    else:
        final_embeddings_m = numpy_array_initial

    return final_embeddings_m, contrastive_embeddings1().detach().cpu().numpy(), contrastive_embeddings2().detach().cpu().numpy()


def embed_dblp():
    print(time.ctime(), '\tLoading data...')
    g1, g2 = pickle.load(open('../data/dblp/networks', 'rb'))
    print(time.ctime(), '\t Size of two networks:', len(g1), len(g2))
    attrs = pickle.load(open('../data/dblp/attrs', 'rb'))
    anchors = dict(json.load(open('../data/dblp/anchors.txt', 'r')))
    print(time.ctime(), '\t # of Anchors:', len(anchors))
    topic = []
    for i in range(len(attrs)):
        v = attrs[i]
        #dblp_1是v[2]
        #dblp_2是v[1]
        topic.append(v[1])
    print(len(topic))
    for seed in [42]:
        for d in [768]:
            # emb_m = word2vec_embed(topic, embed_size=768, initial_embed_path='word2vec_embeddings_dblp_2.pkl')
            # emb_m = my_embed(topic, g1, g2, anchors, batch_size=20, temperature=0.05, epochs=20, learning_rate=0.001,
            #                 between_network_contrastive=True, initial_embed_path='initial_embeddings_dblp_2.pkl')

            #emb_m = pickle.load(open('../emb/word2vec_embeddings_dblp_1.pkl', 'rb'))
            # print(emb_m.shape)
            # print(time.ctime(), '\tNetwork embedding...')
            # emb_g1, emb_g2 = network_embed(g1, g2, anchors, dim=768, batch_size=20,
            #                                 contrastive=True, epochs=50, learning_rate=0.001,
            #                                 initial_embed_path1='initial_embeddings1_dblp_2.pkl', initial_embed_path2='initial_embeddings2_dblp_2.pkl')
            # emb_g1.update(emb_g2)
            # emb_s = np.array([emb_g1[str(i)] for i in range(len(emb_g1))])

            emb_m, emb_g1, emb_g2 = joint_embed(g1, g2, anchors, batch_size=20, temperature=0.05, epochs=20, learning_rate=0.01,
                                                emb_m_contrastive=False,
                                                emb_s_contrastive=True,
                                                initial_embed_emb_m_path='word2vec_embeddings_dblp_2.pkl',
                                                initial_embed_emb_s_path1='initial_embeddings1_dblp_2.pkl',
                                                initial_embed_emb_s_path2='initial_embeddings2_dblp_2.pkl')
            #
            emb_m = (emb_m - np.mean(emb_m, axis=0, keepdims=True)) / np.std(emb_m, axis=0, keepdims=True)
            emb_s = np.vstack([emb_g1, emb_g2])
            emb_s = (emb_s - np.mean(emb_s, axis=0, keepdims=True)) / np.std(emb_s, axis=0, keepdims=True)

            pickle.dump((emb_m, emb_s), open('../emb/emb_dblp_1_joint_initial', 'wb'))


def embed_wd():
    non_chinese_pattern = re.compile(r'[^\u4e00-\u9fa5]+')
    multiple_space_pattern = re.compile(r'\s+')

    def tokenizer_cn(text):
        text = zhconv.convert(text, 'zh-hans').strip()
        text = non_chinese_pattern.sub(' ', text)
        text = multiple_space_pattern.sub(' ', text)
        words = jieba.lcut(text)
        words = [word for word in words if word.strip() != '']
        return words

    def preproc_cn(docs):
        stop_words = pickle.load(open('../data/wd/stop_words_cn.pkl', 'rb'))
        stop_words = set(stop_words)
        docs = [tokenizer_cn(doc) for doc in docs]
        docs = [[word for word in document if word not in stop_words] for document in docs]
        docs = [''.join(doc) for doc in docs]
        return docs

    print(time.ctime(), '\tLoading data...')
    g1, g2 = pickle.load(open('../data/wd/networks', 'rb'))
    print(time.ctime(), '\t Size of two networks:', len(g1), len(g2))
    attrs = pickle.load(open('../data/wd/attrs', 'rb'))
    anchors = dict(json.load(open('../data/wd/anchors.txt', 'r')))
    print(time.ctime(), '\t # of Anchors:', len(anchors))

    docs = []
    for i in range(len(attrs)):
        v = attrs[i]
        text = v[2]
        docs.append(text)
    print('原始文档数量:', len(docs))

    # 对文本进行预处理
    topic = preproc_cn(docs)
    print('预处理后文档数量:', len(topic))
    print('示例文档:', topic[0])
    for seed in [42]:
        for d in [768]:
            # emb_m = word2vec_embed(topic, embed_size=768, initial_embed_path='word2vec_embeddings_wd.pkl')
            # emb_m = my_embed(topic, g1, g2, anchors, batch_size=20, temperature=0.05, epochs=20, learning_rate=0.001,
            #                 between_network_contrastive=True, initial_embed_path='initial_embeddings_wd.pkl', type='cn')
            #
            # emb_m = pickle.load(open('../emb/word2vec_embeddings_wd.pkl', 'rb'))
            # print(emb_m.shape)
            # print(time.ctime(), '\tNetwork embedding...')
            emb_g1, emb_g2 = network_embed(g1, g2, anchors, dim=768, batch_size=20,
                                            contrastive=True, epochs=50, learning_rate=0.001,
                                            initial_embed_path1='initial_embeddings1_wd.pkl', initial_embed_path2='initial_embeddings2_wd.pkl')
            emb_g1.update(emb_g2)
            emb_s = np.array([emb_g1[str(i)] for i in range(len(emb_g1))])

            # emb_m, emb_g1, emb_g2 = joint_embed(
            #     g1, g2, anchors, batch_size=20, temperature=0.05, epochs=20, learning_rate=0.01,
            #     emb_m_contrastive=False,
            #     emb_s_contrastive=True,
            #     initial_embed_emb_m_path='word2vec_embeddings_wd.pkl',
            #     initial_embed_emb_s_path1='initial_embeddings1_wd.pkl',
            #     initial_embed_emb_s_path2='initial_embeddings2_wd.pkl'
            # )
            # #
            # emb_m = (emb_m - np.mean(emb_m, axis=0, keepdims=True)) / np.std(emb_m, axis=0, keepdims=True)
            # emb_s = np.vstack([emb_g1, emb_g2])
            # emb_s = (emb_s - np.mean(emb_s, axis=0, keepdims=True)) / np.std(emb_s, axis=0, keepdims=True)
            #
            # pickle.dump((emb_m, emb_s), open('../emb/emb_wd_joint_initial', 'wb'))


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
inputs = 1
if int(inputs) == 0:
    print('Embedding dataset: dblp')
    embed_dblp()
else:
    print('Embedding dataset: wd')
    embed_wd()