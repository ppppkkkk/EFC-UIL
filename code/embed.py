import json
import tensorflow as tf
from ge import LINE
import torch
from sentence_transformers.ConSERT import ConSERT
from torch.cuda.amp import autocast, GradScaler
import time
import numpy as np
import networkx as nx
import pickle, os
import torch.nn as nn
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


def my_embed(docs, G1, G2, anchors, batch_size=20, temperature=0.1, epochs=20, learning_rate=0.0001,
             in_network_contrastive=True, between_network_contrastive=True, initial_embed_path='initial_embeddings_dblp_1.pkl', final_embed_path_1='final_embeddings_combined_dblp_1_1.pkl'):

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model_name = '../bert-base-uncased'
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

    contrastive_model = ContrastiveLearningModel(numpy_array_initial).to(device)

    optimizer = torch.optim.Adam(contrastive_model.parameters(), lr=learning_rate)
    scaler = GradScaler()

    print(time.ctime(), '\tStarting contrastive learning training...')
    contrastive_model.train()
    dropout_layer = nn.Dropout(p=0.2)
    if in_network_contrastive:
        for epoch in range(epochs):
            print(f"Epoch {epoch + 1} starts")
            epoch_loss = 0.0
            optimizer.zero_grad()

            for i in range(0, len(docs), batch_size):
                print(i)
                batch_docs = docs[i:i + batch_size]
                tokenized = batch_tokenize(model, batch_docs, device)

                # 生成负样本嵌入（从batch中任意取）
                negative_indices = torch.randperm(len(tokenized))  # 随机打乱顺序作为负样本
                neg_docs = [batch_docs[idx] for idx in negative_indices]

                # 对原始句子进行数据增强并作为正样本
                positive_features = tokenized

                # 限制每个特征的长度
                for features in positive_features:
                    for key, value in features.items():
                        features[key] = value[:, :512]

                with autocast():
                    # 生成正样本嵌入
                    pos_out_features = [model.forward(features) for features in positive_features]
                    pos_embeddings = torch.cat([out['sentence_embedding'] for out in pos_out_features]).to(device)
                    pos_embeddings = dropout_layer(pos_embeddings)
                    # 生成负样本嵌入
                    neg_tokenized = batch_tokenize(model, neg_docs, device)
                    neg_out_features = [model.forward(features) for features in neg_tokenized]
                    neg_embeddings = torch.cat([out['sentence_embedding'] for out in neg_out_features]).to(device)
                    neg_embeddings = dropout_layer(neg_embeddings)
                    # 获取当前嵌入
                    current_embeddings = contrastive_model()

                    # 计算对比学习损失
                    loss = contrastive_loss(pos_embeddings, neg_embeddings, current_embeddings[i:i + batch_size],
                                            temperature)

                    epoch_loss += loss.item()

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

                torch.cuda.synchronize()

                del pos_embeddings, neg_embeddings
                torch.cuda.empty_cache()
            print(f'Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(docs)}')

    if os.path.exists(final_embed_path_1):
        # 如果文件存在，直接读取嵌入
        initial_embeddings = pickle.load(open(final_embed_path_1, 'rb'))
        print("Embeddings loaded from final_embed_path_1.")
    else:
        # 如果文件不存在，生成嵌入并保存
        contrastive_model.eval()
        with torch.no_grad():
            initial_embeddings = contrastive_model().cpu().detach().numpy()

        # 保存生成的嵌入到文件
        with open(final_embed_path_1, 'wb') as f:
            pickle.dump(initial_embeddings, f)
        print("final_embed_path_1 generated and saved to file.")

    print(initial_embeddings.shape)
    # 网络间对比学习
    if not nx.is_directed(G1):
        G1 = G1.to_directed()
    if not nx.is_directed(G2):
        G2 = G2.to_directed()

    initial_embeddings_G1 = initial_embeddings[:len(G1.nodes())]
    initial_embeddings_G2 = initial_embeddings[len(G1.nodes()):]

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

                # 添加锚点嵌入
                anchor_embeds1.append(initial_embeddings_G1[node_to_idx_G1[anchor_node1]])
                anchor_embeds2.append(initial_embeddings_G2[node_to_idx_G2[anchor_node2]])

                # 选择 anchor_node2 的所有邻居作为 anchor_node1 的负样本
                neighbors2 = list(G2.neighbors(anchor_node2))
                neg_embeds1.extend([initial_embeddings_G2[node_to_idx_G2[neighbor]] for neighbor in neighbors2])

                # 选择 anchor_node1 的所有邻居作为 anchor_node2 的负样本
                neighbors1 = list(G1.neighbors(anchor_node1))
                neg_embeds2.extend([initial_embeddings_G1[node_to_idx_G1[neighbor]] for neighbor in neighbors1])

            # 分批处理锚点和负样本
            num_batches = len(anchor_embeds1) // batch_size + (1 if len(anchor_embeds1) % batch_size != 0 else 0)

            for batch_idx in range(num_batches):
                batch_start = batch_idx * batch_size
                batch_end = min((batch_idx + 1) * batch_size, len(anchor_embeds1))

                # 将当前批次的嵌入转换为 numpy.ndarray，然后转换为 torch.tensor
                batch_anchor_embeds1 = torch.tensor(np.array(anchor_embeds1[batch_start:batch_end])).float().to(device)
                batch_anchor_embeds2 = torch.tensor(np.array(anchor_embeds2[batch_start:batch_end])).float().to(device)
                batch_neg_embeds1 = torch.tensor(np.array(neg_embeds1[batch_start:batch_end])).float().to(device)
                batch_neg_embeds2 = torch.tensor(np.array(neg_embeds2[batch_start:batch_end])).float().to(device)

                # 使用自定义的对比损失函数
                loss = contrastive_loss(batch_anchor_embeds1, batch_neg_embeds1, batch_anchor_embeds2,
                                        temperature=0.05) + \
                       contrastive_loss(batch_anchor_embeds2, batch_neg_embeds2, batch_anchor_embeds1, temperature=0.05)

                total_loss += loss.item()

                # 批量处理后更新嵌入
                with torch.no_grad():
                    for i, anchor_node1 in enumerate(list(anchors.keys())[batch_start:batch_end]):
                        initial_embeddings_G1[node_to_idx_G1[anchor_node1]] -= loss.item() * 0.001
                        initial_embeddings_G2[node_to_idx_G2[anchors[anchor_node1]]] -= loss.item() * 0.001

            print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(anchors)}")

    print("between_network_contrastive finish")

    final_embeddings = np.concatenate([initial_embeddings_G1, initial_embeddings_G2], axis=0)
    final_embeddings = (final_embeddings - np.mean(final_embeddings, axis=0, keepdims=True)) / np.std(final_embeddings,
                                                                                                      axis=0,
                                                                                                      keepdims=True)

    # 保存最终的嵌入
    pickle.dump(final_embeddings, open('../emb/final_embeddings_combined_dblp_1_2.pkl', 'wb'))

    print(f"Final concatenated embeddings shape: {final_embeddings.shape}")
    return final_embeddings


def cosine_similarity(vec1, vec2):
    return nn.functional.cosine_similarity(vec1.unsqueeze(0), vec2.unsqueeze(0)).item()


def contrastive_loss_network(anchor_embed, positive_embed, negative_embed, temperature=0.05):
    pos_similarity = nn.functional.cosine_similarity(anchor_embed, positive_embed)
    neg_similarity = nn.functional.cosine_similarity(anchor_embed, negative_embed)

    loss = -torch.log(torch.exp(pos_similarity / temperature) / (
                torch.exp(pos_similarity / temperature) + torch.exp(neg_similarity / temperature)))
    return loss.mean()


def network_embed(G1, G2, anchors, dim=768, method="line", order='all', batch_size=20, contrastive=False, epochs=10,
                  initial_embed_path1='initial_embeddings1_dblp_1.pkl', initial_embed_path2='initial_embeddings2_dblp_1.pkl'):
    if not nx.is_directed(G1):
        G1 = G1.to_directed()
    if not nx.is_directed(G2):
        G2 = G2.to_directed()

    G1 = nx.relabel_nodes(G1, lambda x: str(x))
    G2 = nx.relabel_nodes(G2, lambda x: str(x))

    for anchor_node1, anchor_node2 in anchors.items():
        anchor_node1, anchor_node2 = str(anchor_node1), str(anchor_node2)

        # G1 中 A1 的邻居
        neighbors1 = list(G1.neighbors(anchor_node1))
        for neighbor in neighbors1:
            if neighbor in anchors:  # 该邻居是锚节点
                corresponding_neighbor = anchors[neighbor]  # 找到对应 G2 中的锚节点

                # 如果 G2 中 B1 的邻居没有该对应锚节点，则添加一条边
                if corresponding_neighbor not in G2.neighbors(anchor_node2):
                    G2.add_edge(anchor_node2, corresponding_neighbor)
                    G2.add_edge(corresponding_neighbor, anchor_node2)  # 添加双向边

        # G2 中 B1 的邻居
        neighbors2 = list(G2.neighbors(anchor_node2))
        for neighbor in neighbors2:
            if neighbor in anchors.values():  # 该邻居是锚节点
                corresponding_neighbor = list(anchors.keys())[list(anchors.values()).index(neighbor)]  # 找到对应 G1 中的锚节点

                # 如果 G1 中 A1 的邻居没有该对应锚节点，则添加一条边
                if corresponding_neighbor not in G1.neighbors(anchor_node1):
                    G1.add_edge(anchor_node1, corresponding_neighbor)
                    G1.add_edge(corresponding_neighbor, anchor_node1)  # 添加双向边

    print("G1 and G2 densification complete")

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
            print(f"Epoch {epoch + 1}/{epochs}")

            anchor_embeds1, anchor_embeds2, neg_embeds1, neg_embeds2 = [], [], [], []

            for anchor_node1, anchor_node2 in anchors.items():
                anchor_node1, anchor_node2 = str(anchor_node1), str(anchor_node2)

                # 添加锚点嵌入
                anchor_embeds1.append(embeddings1[anchor_node1])
                anchor_embeds2.append(embeddings2[anchor_node2])

                # 选择 anchor_node2 的所有邻居作为 anchor_node1 的负样本
                neighbors2 = list(G2.neighbors(anchor_node2))
                neg_embeds1.extend([embeddings2[neighbor] for neighbor in neighbors2])

                # 选择 anchor_node1 的所有邻居作为 anchor_node2 的负样本
                neighbors1 = list(G1.neighbors(anchor_node1))
                neg_embeds2.extend([embeddings1[neighbor] for neighbor in neighbors1])

            # 分批处理锚点和负样本
            num_batches = len(anchor_embeds1) // batch_size + (1 if len(anchor_embeds1) % batch_size != 0 else 0)

            for batch_idx in range(num_batches):
                batch_start = batch_idx * batch_size
                batch_end = min((batch_idx + 1) * batch_size, len(anchor_embeds1))

                # 将当前批次的嵌入转换为 numpy.ndarray，然后转换为 torch.tensor
                batch_anchor_embeds1 = torch.tensor(np.array(anchor_embeds1[batch_start:batch_end])).float().to(device)
                batch_anchor_embeds2 = torch.tensor(np.array(anchor_embeds2[batch_start:batch_end])).float().to(device)
                batch_neg_embeds1 = torch.tensor(np.array(neg_embeds1[batch_start:batch_end])).float().to(device)
                batch_neg_embeds2 = torch.tensor(np.array(neg_embeds2[batch_start:batch_end])).float().to(device)

                # 使用自定义的对比损失函数
                loss = contrastive_loss(batch_anchor_embeds1, batch_neg_embeds1, batch_anchor_embeds2, temperature=0.05) + \
                       contrastive_loss(batch_anchor_embeds2, batch_neg_embeds2, batch_anchor_embeds1, temperature=0.05)

                total_loss += loss.item()

                # 批量处理后更新嵌入
                with torch.no_grad():
                    for i, anchor_node1 in enumerate(list(anchors.keys())[batch_start:batch_end]):
                        embeddings1[str(anchor_node1)] -= loss.item() * 0.001
                        embeddings2[str(anchors[anchor_node1])] -= loss.item() * 0.001

            print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(anchors)}")

    return embeddings1, embeddings2


def joint_embed(docs, G1, G2, anchors, batch_size=20, dim=768, temperature=0.1, epochs=20, learning_rate=0.0001,
             emb_m_in_network_contrastive=True, emb_m_between_network_contrastive=True, emb_s_contrastive=True, initial_embed_emb_m_path='initial_embeddings_dblp_1.pkl', initial_embed_emb_s_path1='initial_embeddings1_dblp_1.pkl', initial_embed_emb_s_path2='initial_embeddings2_dblp_1.pkl'):

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if os.path.exists(initial_embed_emb_m_path):
        with open(initial_embed_emb_m_path, 'rb') as f:
            numpy_array_initial = pickle.load(f)
        print(f"Loaded initial embeddings from {initial_embed_emb_m_path}")

    contrastive_model = ContrastiveLearningModel(numpy_array_initial).to(device)

    optimizer = torch.optim.Adam(contrastive_model.parameters(), lr=learning_rate)
    scaler = GradScaler()

    print(time.ctime(), '\tStarting contrastive learning training...')
    contrastive_model.train()
    dropout_layer = nn.Dropout(p=0.2)




def embed_dblp():
    print(time.ctime(), '\tLoading data...')
    g1, g2 = pickle.load(open('../data/dblp/networks', 'rb'))
    print(time.ctime(), '\t Size of two networks:', len(g1), len(g2))
    attrs = pickle.load(open('../data/dblp/attrs', 'rb'))
    anchors = dict(json.load(open('../data/dblp/anchors.txt', 'r')))
    print(anchors)
    print(time.ctime(), '\t # of Anchors:', len(anchors))
    topic = []
    print(len(attrs))
    print(attrs[0])
    for i in range(len(attrs)):
        v = attrs[i]
        topic.append(v[2])
    print(topic)
    for seed in [42]:
        for d in [768]:
            print(time.ctime(), '\tMy level attributes embedding...')
            # emb_m = my_embed(topic, g1, g2, anchors, batch_size=10, temperature=0.1, epochs=20, learning_rate=0.0001,
            #         in_network_contrastive=False, between_network_contrastive=True)
            # print(emb_m.shape)
            # print(time.ctime(), '\tNetwork embedding...')
            emb_g1, emb_g2 = network_embed(g1, g2, anchors, dim=768,
                                            method="line", order='all', contrastive=True, epochs=50)
            emb_g1.update(emb_g2)
            emb_s = np.array([emb_g1[str(i)] for i in range(len(emb_g1))])
            #
            emb_s = (emb_s - np.mean(emb_s, axis=0, keepdims=True)) / np.std(emb_s, axis=0, keepdims=True)
            # emb_m = (emb_m - np.mean(emb_m, axis=0, keepdims=True) / np.std(emb_m, axis=0, keepdims=True))
            #
            # # Saving embeddings
            pickle.dump(emb_s, open('../emb/emb_s_dblp_1_con_new_jiabian', 'wb'))
            # pickle.dump((emb_m, emb_s), open('../emb/emb_s_dblp_2_con', 'wb'))




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