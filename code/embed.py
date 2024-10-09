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
from gensim.corpora import Dictionary
from gensim import utils,models
from sentence_transformers import SentenceTransformer

stop_words = pickle.load(open('../data/wd/stop_words_cn.pkl', 'rb'))
tf.compat.v1.disable_v2_behavior()

def word2vec_embed(docs, embed_size=768, window=5, min_count=1, workers=4,
                   initial_embed_path='.pkl'):
    # 先检查是否存在已经保存的嵌入
    if os.path.exists(initial_embed_path):
        with open(initial_embed_path, 'rb') as f:
            embeddings = pickle.load(f)
        print(f"Loaded word2vec embeddings from {initial_embed_path}")
        return embeddings

    docs_part1 = docs[:9714]
    docs_part2 = docs[-9526:]

    # tokenized_docs_part1 = [doc.split() for doc in docs_part1]
    # tokenized_docs_part2 = [doc.split() for doc in docs_part2]

    print("Training Word2Vec model...")
    all_tokenized_docs = docs_part1 + docs_part2
    word2vec_model = Word2Vec(sentences=all_tokenized_docs, vector_size=embed_size, window=window, min_count=min_count,
                              workers=workers)

    def generate_embeddings(tokenized_docs):
        embeddings = []
        for doc in tokenized_docs:
            valid_words = [word for word in doc if word in word2vec_model.wv]

            if len(valid_words) > 0:
                doc_embedding = np.mean([word2vec_model.wv[word] for word in valid_words], axis=0)
                embeddings.append(doc_embedding)
            else:
                embeddings.append(np.zeros(embed_size))
        return np.array(embeddings)

    print("Generating document embeddings for the first documents...")
    embeddings_part1 = generate_embeddings(docs_part1)

    print("Generating document embeddings for the last documents...")
    embeddings_part2 = generate_embeddings(docs_part2)
    embeddings = np.concatenate((embeddings_part1, embeddings_part2), axis=0)

    # 保存嵌入到文件
    with open(initial_embed_path, 'wb') as f:
        pickle.dump(embeddings, f)
    print(f"Word2Vec embeddings saved to {initial_embed_path}")

    return embeddings


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

    embeddings1 = np.array([embeddings1[node] for node in G1.nodes()])
    embeddings2 = np.array([embeddings2[node] for node in G2.nodes()])

    return embeddings1, embeddings2


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
        # dblp_1是v[2]
        # dblp_2是v[1]
        topic.append(v[1])
    for seed in [42]:
        for d in [768]:
            word2vec_embed(topic, embed_size=768, initial_embed_path='word2vec_embeddings_dblp_2_test.pkl')


def topic_embed(docs, dim=768):
    dict_ = Dictionary(docs)

    # Filter out words that occur less than 10 documents, or more than 50% of the documents.
    dict_.filter_extremes(no_below=10, no_above=0.5)

    corpus = [dict_.doc2bow(doc) for doc in docs]

    print('Number of unique tokens: %d' % len(dict_))
    print('Number of documents: %d' % len(corpus))

    # Train LDA model.
    # Make a index to word dictionary.
    _ = dict_[0]  # 这里可能是bug，必须要调用一下dictionary才能确保被加载

    id2word = dict_.id2token

    print(time.ctime(), '\tLearning topic model...')
    model = models.LdaModel(
        corpus=corpus,
        id2word=id2word,
        chunksize=256,
        passes=4,
        # alpha='asymmetric',  # 1/topic_num
        # eta='asymmetric',    # 1/topic_num
        # distributed=True,
        iterations=100,
        num_topics=dim,
        eval_every=1,
        minimum_probability=1e-13,
        random_state=0)

    topic_dist = model.get_document_topics(corpus)
    # 如果emb的维度低于dimension,则说明部分话题维度太小被省略了，则需要进行填充
    embed = []
    for i in range(len(corpus)):
        emb = []
        topic_i = topic_dist[i]

        if len(topic_i) < dim:
            topic_i = dict(topic_i)
            for j in range(dim):
                if j in topic_i.keys():
                    emb.append(topic_i[j])
                else:
                    emb.append(1e-13)
        else:
            emb = np.array(topic_i, dtype=np.float64)[:, 1]

        embed.append(emb)

    embed = np.array(embed)

    return embed


def sbert_embed_chinese(docs):
    model = SentenceTransformer('uer/sbert-base-chinese-nli', cache_folder='./hf_cache')
    print("Generating embeddings using SBERT...")
    embeddings = model.encode(docs, show_progress_bar=True)
    return np.array(embeddings)


# def embed_wd():
#     # 简化的文本清洗步骤
#     def clean_text(text):
#         # 将繁体中文转换为简体中文
#         text = zhconv.convert(text, 'zh-hans').strip()
#         # 移除不必要的特殊符号，但保留中文、英文和数字
#         text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\s]', '', text)
#         text = re.sub(r'\s+', ' ', text).strip()
#         return text
#
#     print(time.ctime(), '\tLoading data...')
#     try:
#         g1, g2 = pickle.load(open('../data/wd/networks', 'rb'))
#         print(time.ctime(), '\t Size of two networks:', len(g1), len(g2))
#         attrs = pickle.load(open('../data/wd/attrs', 'rb'))
#         anchors = dict(json.load(open('../data/wd/anchors.txt', 'r')))
#         print(time.ctime(), '\t # of Anchors:', len(anchors))
#     except FileNotFoundError as e:
#         print(f"Error loading data: {e}")
#         return
#
#     docs = [attrs[i][2] for i in range(len(attrs))]
#     print('原始文档数量:', len(docs))
#
#     # 对文本进行简单的清洗
#     cleaned_docs = [clean_text(doc) for doc in docs]
#     print('清洗后文档数量:', len(cleaned_docs))
#     print('示例文档:', cleaned_docs[0])
#     print('示例文档:', cleaned_docs[min(10000, len(cleaned_docs)-1)])
#
#     # 使用 SBERT 生成嵌入
#     attr_a = sbert_embed_chinese(cleaned_docs)
#     initial_embed_path = 'sbert_embeddings_wd_test.pkl'
#
#     # 确保保存路径存在
#     directory = os.path.dirname(initial_embed_path)
#     if directory:
#         os.makedirs(directory, exist_ok=True)
#
#     # 保存嵌入到文件
#     with open(initial_embed_path, 'wb') as f:
#         pickle.dump(attr_a, f)
#     print(f"Embeddings saved to {initial_embed_path}")


from sklearn.feature_extraction.text import CountVectorizer


def embed_wd_bow():
    # 简化的文本清洗步骤
    def clean_text(text):
        # 将繁体中文转换为简体中文
        text = zhconv.convert(text, 'zh-hans').strip()
        # 移除不必要的特殊符号，但保留中文、英文和数字
        text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    print(time.ctime(), '\tLoading data...')
    try:
        g1, g2 = pickle.load(open('../data/wd/networks', 'rb'))
        print(time.ctime(), '\t Size of two networks:', len(g1), len(g2))
        attrs = pickle.load(open('../data/wd/attrs', 'rb'))
        anchors = dict(json.load(open('../data/wd/anchors.txt', 'r')))
        print(time.ctime(), '\t # of Anchors:', len(anchors))
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        return

    docs = [attrs[i][2] for i in range(len(attrs))]
    print('原始文档数量:', len(docs))

    # 对文本进行简单的清洗
    cleaned_docs = [clean_text(doc) for doc in docs]
    print('清洗后文档数量:', len(cleaned_docs))
    print('示例文档:', cleaned_docs[0])
    print('示例文档:', cleaned_docs[min(10000, len(cleaned_docs) - 1)])

    # 使用词袋模型生成嵌入
    vectorizer = CountVectorizer(max_features=768)  # 设置最大特征数，可以根据需要调整
    bow_matrix = vectorizer.fit_transform(cleaned_docs).toarray()

    print('词袋嵌入矩阵的形状:', bow_matrix.shape)

    initial_embed_path = 'bow_embeddings_wd_test.pkl'

    # 确保保存路径存在
    directory = os.path.dirname(initial_embed_path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    # 保存词袋嵌入到文件
    with open(initial_embed_path, 'wb') as f:
        pickle.dump(bow_matrix, f)
    print(f"Bag-of-Words embeddings saved to {initial_embed_path}")


def embed_wd():
    import jieba, zhconv, re
    from gensim.corpora import WikiCorpus

    def tokenizer_cn(text, token_min_len=10, token_max_len=100, lower=False):
        text = zhconv.convert(text, 'zh-hans').strip()  # Standardize to simple Chinese
        text = p.sub('', text)
        return jieba.lcut(text)

    def preproc_cn(docs, min_len=2, max_len=15):
        docs = [tokenizer_cn(doc) for doc in docs]
        # Removing Stop words
        stop_words = pickle.load(open('../data/wd/stop_words_cn.pkl', 'rb'))
        stop_words = set(stop_words)
        docs = [[word for word in document if word not in stop_words] for document in docs]
        return docs

    def topic_embed_cn(docs, dim=768):
        return topic_embed(docs, dim=dim)

    p = re.compile('[^\u4e00-\u9fa5]')

    print(time.ctime(), '\tLoading data...')
    g1, g2 = pickle.load(open('../data/wd/networks', 'rb'))
    print(time.ctime(), '\t Size of two networks:', len(g1), len(g2))
    attrs = pickle.load(open('../data/wd/attrs', 'rb'))
    char_corpus, word_corpus, topic_corpus = [], [], []
    for i in range(len(attrs)):
        v = attrs[i]
        topic_corpus.append(v[2])
        # The index number is the node id of users in the two networks.

    print(time.ctime(), '\tPreprocessing...')
    topic_corpus = preproc_cn(topic_corpus)

    for seed in range(1):
        for d in [768]:
            print(time.ctime(), '\tTopic level attributes embedding...')
            emb_t = word2vec_embed(topic_corpus, embed_size=d)

            # Standardization

            emb_t = (emb_t - np.mean(emb_t, axis=0, keepdims=True)) / np.std(emb_t, axis=0, keepdims=True)

            # Saving embeddings
            pickle.dump((emb_t), open('../emb/emb_wd_seed_{}_dim_{}'.format(seed, d), 'wb'))



os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
inputs = 0
if int(inputs) == 1:
    print('Embedding dataset: dblp')
    embed_dblp()
else:
    print('Embedding dataset: wd')
    embed_wd()