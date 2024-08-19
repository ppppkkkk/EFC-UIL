# -*- coding: utf-8 -*-
import random

import numpy as np
import networkx as nx
import json, pickle, os, time
from gensim import utils, models
from gensim.models import Word2Vec
from gensim.corpora import Dictionary
import tensorflow as tf
import torch.nn as nn
from torch_geometric.graphgym import optim

from sentence_transformers.util import batch_to_device

tf.compat.v1.disable_v2_behavior()
from tf_slim.layers import fully_connected
from ge import Struc2Vec, LINE
import nltk
from sentence_transformers.ConSERT import ConSERT
import torch
from torch.utils.data import DataLoader
import math
from sentence_transformers import InputExample
from sklearn.decomposition import PCA
from torch.nn import functional as F

# 3-layer bert encoder

def contrastive_loss(anchor, positive_or_negative, temperature):
    similarity = torch.nn.functional.cosine_similarity(anchor, positive_or_negative, dim=-1)
    loss = -torch.log(torch.nn.functional.softmax(similarity / temperature, dim=-1))
    return loss.mean()

def batch_tokenize(model, docs, device):
    tokenized = [model.tokenize([doc]) for doc in docs]
    tokenized = [{key: value.to(device) for key, value in doc.items()} for doc in tokenized]
    return tokenized

def my_embed(docs, dim=100, batch_size=1024, temperature=0.05, epochs=20, learning_rate=1e-5):
    model_name = 'B:/bert-base-uncased'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ConSERT(model_name, device=device, cutoff_rate=0.12, close_dropout=True)
    model.__setattr__("max_seq_length", 64)
    print(time.ctime(), '\tLearning my vectors...')
    long_text = docs

    model.to(device)
    model.train()

    linear_layer = nn.Linear(768, dim).to(device)
    optimizer = torch.optim.Adam(linear_layer.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        epoch_loss = 0.0
        for i in range(0, len(long_text), batch_size):
            batch_docs = long_text[i:i + batch_size]
            tokenized = batch_tokenize(model, batch_docs, device)

            positive_features = batch_tokenize(model, batch_docs, device)
            negative_features = [model.data_augment(doc) for doc in tokenized]

            with torch.no_grad():
                pos_out_features = [model.forward(features) for features in positive_features]
                neg_out_features = [model.forward(features) for features in negative_features]

            pos_embeddings = torch.cat([out['sentence_embedding'] for out in pos_out_features])
            neg_embeddings = torch.cat([out['sentence_embedding'] for out in neg_out_features])
            pos_vector = linear_layer(pos_embeddings)
            neg_vector = linear_layer(neg_embeddings)

            # 对比学习损失
            pos_loss = contrastive_loss(pos_vector, pos_vector, temperature)
            neg_loss = contrastive_loss(pos_vector, neg_vector, temperature)
            loss = pos_loss + neg_loss
            epoch_loss += loss.item()

            # 反向传播与优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            del pos_embeddings, neg_embeddings, pos_vector, neg_vector
            torch.cuda.empty_cache()

        print(f'Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(long_text)}')

    # 最终嵌入生成
    embed = []
    model.eval()
    for i in range(0, len(long_text), batch_size):
        batch_docs = long_text[i:i + batch_size]
        tokenized = batch_tokenize(model, batch_docs, device)
        with torch.no_grad():
            out_features = [model.forward(features) for features in tokenized]
            embeddings = torch.cat([out['sentence_embedding'] for out in out_features])
            reduced_vector = linear_layer(embeddings)
            embed.append(reduced_vector)
            del embeddings, reduced_vector
            torch.cuda.empty_cache()

    numpy_array = torch.cat(embed, dim=0).cpu().detach().numpy()
    print(numpy_array.shape)
    return numpy_array


def my_embed_cn(docs, dim=100):
    model_name = 'B:/chinese-roberta-wwm-ext'
    device = torch.device('cuda')
    model = ConSERT(model_name, device=device, cutoff_rate=0.12, close_dropout=True)
    model.__setattr__("max_seq_length", 64)
    print(time.ctime(), '\tLearning my vectors...')
    long_text = docs

    device = torch.device('cuda')
    model.to(device)
    model.eval()

    embed = []
    linear_layer = nn.Linear(768, dim).to(device)
    #print(docs)
    for doc in long_text:
        # 将文档转换为模型可以接受的格式
        print(doc)
        tokenized = model.tokenize([doc])
        input_example = InputExample(texts=tokenized, label=0)
        sentence_features, _ = model.smart_batching_collate([input_example])
        #print(sentence_features)
        with torch.no_grad():
            # 使用增强后的features进行编码
            out_features = model.forward(sentence_features[0])

            embeddings = out_features['sentence_embedding']
            #print(out_features)
            reduced_vector = linear_layer(embeddings)

        embed.append(reduced_vector)
        del embeddings, reduced_vector
        torch.cuda.empty_cache()

    numpy_array = np.array([tensor.cpu().detach().numpy() for tensor in embed])
    numpy_array = np.squeeze(numpy_array, axis=1)
    print(numpy_array.shape)
    return numpy_array


## only bert encoder
def my_embed_bert(docs, dim=100):
    print(time.ctime(), '\tLearning my vectors...')
    #docs是所有的句子，每个人对应的文章标题为一行，一行可能有多个标题，每个标题用\n隔开，为一整个字符串
    long_text = docs

    model_name = 'B:/bert-base-uncased'
    device = "cuda:0"
    model = ConSERT(model_name, device=device, cutoff_rate=0.0, close_dropout=True)
    model.__setattr__("max_seq_length", 64)
    embed = []
    #print(long_text)
    linear_layer = nn.Linear(768, dim).to(device)
    for doc in long_text:
        embedding = model.encode(doc, batch_size=64, show_progress_bar=False,
                                     convert_to_numpy=False)
        reduced_vector = linear_layer(embedding)
        embed.append(reduced_vector)
        del embedding, reduced_vector
        torch.cuda.empty_cache()

    numpy_array = np.array([tensor.cpu().detach().numpy() for tensor in embed])
    print(numpy_array.shape)
    return numpy_array


def contrastive_loss_network(anchor, positive, negative, temperature=0.05):
    pos_similarity = F.cosine_similarity(anchor, positive, dim=-1)
    neg_similarity = F.cosine_similarity(anchor, negative, dim=-1)
    pos_loss = -torch.log(F.softmax(pos_similarity / temperature, dim=-1))
    neg_loss = -torch.log(1 - F.softmax(neg_similarity / temperature, dim=-1))
    return (pos_loss + neg_loss).mean()


def network_embed(G1, G2, anchors, dim=16, method="line", order='all', contrastive=True, epochs=20):
    if not nx.is_directed(G1):
        G1 = G1.to_directed()
    if not nx.is_directed(G2):
        G2 = G2.to_directed()

    G1 = nx.relabel_nodes(G1, lambda x: str(x))
    G2 = nx.relabel_nodes(G2, lambda x: str(x))

    if method == 'line':
        model1 = LINE(G1, embedding_size=dim, order=order)
        model2 = LINE(G2, embedding_size=dim, order=order)

        model1.train(batch_size=1024, epochs=50, verbose=2)
        model2.train(batch_size=1024, epochs=50, verbose=2)

        embeddings1 = model1.get_embeddings()
        embeddings2 = model2.get_embeddings()

        if contrastive:
            for epoch in range(epochs):
                total_loss = 0.0
                for anchor_node1, anchor_node2 in anchors.items():
                    anchor_node1 = str(anchor_node1)
                    anchor_node2 = str(anchor_node2)

                    # 正样本对: 锚节点在两个网络中的对应节点
                    anchor_embed1 = torch.tensor(embeddings1[anchor_node1]).float()
                    anchor_embed2 = torch.tensor(embeddings2[anchor_node2]).float()

                    # 负样本对: 随机选择其他节点
                    neighbors1 = list(G1.neighbors(anchor_node1))
                    neighbors2 = list(G2.neighbors(anchor_node2))

                    if neighbors1:
                        neg_node1 = random.choice(neighbors1)
                        neg_embed1 = torch.tensor(embeddings1[str(neg_node1)]).float()

                    if neighbors2:
                        neg_node2 = random.choice(neighbors2)
                        neg_embed2 = torch.tensor(embeddings2[str(neg_node2)]).float()

                    loss = contrastive_loss_network(anchor_embed1, anchor_embed2, neg_embed1, temperature=0.05) + \
                           contrastive_loss_network(anchor_embed2, anchor_embed1, neg_embed2, temperature=0.05)
                    total_loss += loss.item()

                    with torch.no_grad():
                        embeddings1[anchor_node1] -= loss.item() * 0.001
                        embeddings2[anchor_node2] -= loss.item() * 0.001

                print(f'Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(anchors)}')

        return embeddings1, embeddings2

    else:
        raise NotImplementedError("Network embedding method: %s not implemented." % method)

# dblp的数据库内容基本为中文的版本
def embed_dblp():
    # 可用的dblp数据库，即我所用的内容
    print(time.ctime(), '\tLoading data...')
    g1, g2 = pickle.load(open('../data/dblp/networks', 'rb'))
    print(time.ctime(), '\t Size of two networks:', len(g1), len(g2))
    attrs = pickle.load(open('../data/dblp/attrs', 'rb'))
    anchors = dict(json.load(open('../data/dblp/anchors.txt', 'r')))  # 加载锚节点数据
    char_corpus, word_corpus, topic_corpus = [], [], []
    topic = []
    print(len(attrs))
    for i in range(len(attrs)):

        # word_corpus是学校的名字+地址
        # char_corpus是姓名
        # topic_corpus是文章的题目，放在一起以\n分割开来

        v = attrs[i]
        topic_corpus.append(v[2])

        # The index number is the node id of users in the two networks.

    for seed in range(3):
        for d in [100]:
            print(time.ctime(), '\tWord level attributes embedding...')
            emb_m = my_embed(topic_corpus, dim=d)

            print(time.ctime(), '\tNetwork 1 embedding...')
            emb_g1, emb_g2 = network_embed(g1, g2, anchors, method='line', dim=d, contrastive=True)

            # 合并嵌入
            emb_g1.update(emb_g2)
            emb_s = np.array([emb_g1[str(i)] for i in range(len(emb_g1))])
            # 正则
            emb_s = (emb_s - np.mean(emb_s, axis=0, keepdims=True)) / np.std(emb_s, axis=0, keepdims=True)
            # m为多加的一行
            emb_m = (emb_m - np.mean(emb_m, axis=0, keepdims=True) / np.std(emb_m, axis=0, keepdims=True))
            # Saving embeddings
            pickle.dump((emb_m, emb_s),open('../emb/emb_dblp_seed_{}_dim_{}'.format(seed, d), 'wb'))



#用于wd的数据库，里面的每个embed函数为中文的版本
def embed_wd(pypinyin=None):
    import jieba, zhconv, re
    from pypinyin import lazy_pinyin
    from gensim.models.word2vec import LineSentence
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

    def clean_text_cn(text, lower=False):
        # 字符简化：将繁体中文转换为简体中文
        text = zhconv.convert(text, 'zh-hans').strip()

        text = p.sub('', text)

        # 是否转换为小写
        if lower:
            text = text.lower()

        return text

    def process_wiki(inp, outp, dct):
        _ = dct[0]
        output = open(outp, 'w', encoding='utf-8')
        wiki = WikiCorpus(inp, processes=os.cpu_count() - 2,
                          dictionary=dct, article_min_tokens=10,
                          lower=False)  # It takes about 16 minutes by 10 core cpu.
        count = 0
        for words in wiki.get_texts():
            words = [" ".join(tokenizer_cn(w)) for w in words]
            output.write(' '.join(words) + '\n')
            count += 1
            if count % 10000 == 0:
                print('Finished %d-67' % count // 10000)
        output.close()

    def topic_embed_cn(docs, dim=100):
        return my_embed_cn(docs, dim)

    p = re.compile('[^\u4e00-\u9fa5]')
    ex_corpus_xml = '../data/wd/zhwiki-latest-pages-articles.xml.bz2'
    ex_corpus_fname = '../data/wd/zhwiki_corpus'

    print(time.ctime(), '\tLoading data...')
    g1, g2 = pickle.load(open('../data/wd/networks', 'rb'))
    print(g1)
    print(time.ctime(), '\t Size of two networks:', len(g1), len(g2))
    attrs = pickle.load(open('../data/wd/attrs', 'rb'))
    char_corpus, word_corpus, topic_corpus = [], [], []
    for i in range(len(attrs)):
        v = attrs[i]
        char_corpus.append(v[0])
        # word_corpus.append(v[1])
        topic_corpus.append(v[2])
        # The index number is the node id of users in the two networks.

    print(time.ctime(), '\tPreprocessing...')
    # word_corpus = preproc_cn(word_corpus)
    # topic_corpus = preproc_cn(topic_corpus)
    for seed in range(3):
        for d in [100]:

            print(time.ctime(), '\tTopic level attributes embedding...')
            emb_t = topic_embed_cn(topic_corpus, dim=d)

            print(time.ctime(), '\tNetwork 1-1 embedding...')
            emb_g1 = network_embed(g1, method='line', dim=d)

            print(time.ctime(), '\tNetwork 1-2 embedding...')
            emb_g2 = network_embed(g2, method='line', dim=d)
            emb_g1.update(emb_g2)
            emb_s = np.array([emb_g1[str(i)] for i in range(len(emb_g1))])

            # Standardization
            emb_t = (emb_t - np.mean(emb_t, axis=0, keepdims=True)) / np.std(emb_t, axis=0, keepdims=True)
            emb_s = (emb_s - np.mean(emb_s, axis=0, keepdims=True)) / np.std(emb_s, axis=0, keepdims=True)

            # Saving embeddings
            pickle.dump((emb_t,emb_s),open('../emb/emb_wd_seed_{}_dim_{}'.format(seed,d),'wb'))

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
#inputs = input('Selecte the dataset(0/1): (0为dblp数据库，可用，1为微博数据库中文的)')
inputs = 0
if int(inputs) == 0:
    print('Embedding dataset: dblp')
    embed_dblp()
    # 可以用这个，英文的科研数据库
else:
    print('Embedding dataset: wd')
    # 带中文的应该是微博数据库
    embed_wd()
