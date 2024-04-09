# -*- coding: utf-8 -*-
import numpy as np
import networkx as nx
import json, pickle, os, time
from gensim import utils, models
from gensim.models import Word2Vec
from gensim.corpora import Dictionary
import tensorflow as tf
import torch.nn as nn
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
def preproc(docs, min_len=2, max_len=15):
    for i in range(len(docs)):
        docs[i] = [token for token in
                   utils.tokenize(docs[i],
                                  lower=True,
                                  deacc=True,
                                  errors='ignore')
                   if min_len <= len(token) <= max_len]

    from nltk.stem.wordnet import WordNetLemmatizer
    lemmatizer = WordNetLemmatizer()
    docs = [[lemmatizer.lemmatize(token) for token in doc] for doc in docs]
    # e.g. years->year, models->model, not including: modeling->modeling

    # NLTK Stop words
    from nltk.corpus import stopwords
    stop_words = stopwords.words('english')
    stop_words.extend(['from', 'subject', 're', 'edu', 'use'])
    stop_words = set(stop_words)

    docs = [[word for word in document if word not in stop_words] for document in docs]

    # Build the bigram and trigram models
    bigram = models.Phrases(docs, min_count=1, threshold=0.1)  # higher threshold fewer phrases.
    trigram = models.Phrases(bigram[docs], threshold=0.1)

    # Get a sentence clubbed as a trigram/bigram
    bigram_mod = models.phrases.Phraser(bigram)
    trigram_mod = models.phrases.Phraser(trigram)

    # Add bigrams and trigrams to docs.
    docs = [bigram_mod[doc] for doc in docs]
    docs = [trigram_mod[bigram_mod[doc]] for doc in docs]

    return docs


def topic_embed(docs, dim=32):
    dict_ = Dictionary(docs)
    # 用的Lda模型
    # Filter out words that occur less than 10 documents, or more than 50% of the documents.
    dict_.filter_extremes(no_below=10, no_above=0.5)
    corpus = [dict_.doc2bow(doc) for doc in docs]
    # print(dict_)
    print('Number of unique tokens: %d' % len(dict_))
    print('Number of documents: %d' % len(corpus))
    # print(corpus)

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
    # print(type(model))
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


# 3-layer bert encoder

def my_embed(docs, dim=100):
    model_name = 'A:/bert-base-uncased'
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
        tokenized = model.tokenize([doc])
        sentence_features = model.data_augment(tokenized)

        with torch.no_grad():
            # 使用增强后的features进行编码
            out_features = model.forward(sentence_features)
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

    model_name = 'A:/bert-base-uncased'
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





def char_embed(docs, dim=16):
    punc = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
    docs = [[w for w in list(doc) if
             not w.isnumeric()
             and not w.isspace()
             and not w in punc] for doc in docs]

    # Build the bigram and trigram models
    bigram = models.Phrases(docs, min_count=1, threshold=0.1)  # higher threshold fewer phrases.
    trigram = models.Phrases(bigram[docs], threshold=0.1)

    # Get a sentence clubbed as a trigram/bigram
    bigram_mod = models.phrases.Phraser(bigram)
    trigram_mod = models.phrases.Phraser(trigram)

    docs = [bigram_mod[doc] for doc in docs]
    docs = [trigram_mod[bigram_mod[doc]] for doc in docs]

    dict_ = Dictionary(docs)

    docs = [dict_.doc2bow(doc) for doc in docs]

    data = np.zeros((len(docs), len(dict_)))
    for n, values in enumerate(docs):
        for idx, value in values:
            data[n][idx] = value

    # from tensorflow.contrib.layers import dropout
    # keep_prob = 0.7

    g = tf.compat.v1.get_default_graph()
    # is_training = tf.placeholder_with_default(False,shape=(),name='is_training')
    X = tf.compat.v1.placeholder(tf.float64, shape=[None, data.shape[1]])
    # X_drop = dropout(X,keep_prob,is_training) #考虑引入噪音
    # 考虑引入权重绑定
    hidden = fully_connected(X, dim, activation_fn=None)
    outputs = fully_connected(hidden, data.shape[1], activation_fn=None)

    loss = tf.reduce_mean(tf.square(outputs - X))
    train_op = tf.compat.v1.train.AdamOptimizer(0.01).minimize(loss)

    with tf.compat.v1.Session(graph=g, config=config) as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        for i in range(50):
            _, embed, loss_ = sess.run([train_op, hidden, loss], feed_dict={X: data})
            if i % 10 == 0:
                # print(loss_)
                pass

    return embed


def network_embed(G, dim=16, l_walks=6, n_walks=32, method="line", order='all', workers=os.cpu_count() - 1):
    if not nx.is_directed(G):
        G = G.to_directed()

    G = nx.relabel_nodes(G, lambda x: str(x))

    if method == 'struc2vec':
        model = Struc2Vec(G, walk_length=l_walks, num_walks=n_walks, workers=workers, verbose=10,
                          opt3_num_layers=4, temp_path='./temp_struc2vec_seperated/')
        model.train(embed_size=dim)
        embeddings = model.get_embeddings()

    elif method == 'line':
        print(dim)
        model = LINE(G, embedding_size=dim, order=order)
        model.train(batch_size=1024, epochs=50, verbose=2)
        embeddings = model.get_embeddings()

    else:
        raise NotImplementedError("Network embedding method: %s not implemented." % method)

    return embeddings


# dblp的数据库内容基本为中文的版本
def embed_dblp():
    # 可用的dblp数据库，即我所用的内容
    print(time.ctime(), '\tLoading data...')
    g1, g2 = pickle.load(open('../data/dblp/networks', 'rb'))
    print(time.ctime(), '\t Size of two networks:', len(g1), len(g2))
    attrs = pickle.load(open('../data/dblp/attrs', 'rb'))
    char_corpus, word_corpus, topic_corpus = [], [], []
    topic = []
    print(len(attrs))
    for i in range(len(attrs)):

        # word_corpus是学校的名字+地址
        # char_corpus是姓名
        # topic_corpus是文章的题目，放在一起以\n分割开来

        v = attrs[i]
        char_corpus.append(v[0])
        #word_corpus.append(v[1])
        topic_corpus.append(v[1])

        #topic.append(v[1])

        # The index number is the node id of users in the two networks.

    #print(topic)
    topic_corpus = preproc(topic_corpus)
    for seed in range(3):
        for d in [100]:
            print(time.ctime(), '\tCharacter level attributes embedding...')
            emb_c = char_embed(char_corpus, dim=d)

            #print(time.ctime(), '\tWord level attributes embedding...')
            print(time.ctime(), '\tMy level attributes embedding...')
            #emb_m = my_embed(topic, dim=d)
            #emb_w = word_embed(word_corpus, lamb=0.1, dim=d, ave_neighbors=True, g1=g1, g2=g2)

            #print(time.ctime(), '\tTopic level attributes embedding...')
            emb_t = topic_embed(topic_corpus, dim=d)
            print(time.ctime(), '\tNetwork 1 embedding...')
            emb_g1 = network_embed(g1, method='line', dim=d)

            print(time.ctime(), '\tNetwork 2 embedding...')
            emb_g2 = network_embed(g2, method='line', dim=d)
            emb_g1.update(emb_g2)
            emb_s = np.array([emb_g1[str(i)] for i in range(len(emb_g1))])
            # 正则化
            #emb_c = (emb_c - np.mean(emb_c, axis=0, keepdims=True)) / np.std(emb_c, axis=0, keepdims=True)
            #emb_w = (emb_w - np.mean(emb_w, axis=0, keepdims=True)) / np.std(emb_w, axis=0, keepdims=True)
            emb_t = (emb_t - np.mean(emb_t, axis=0, keepdims=True)) / np.std(emb_t, axis=0, keepdims=True)
            emb_s = (emb_s - np.mean(emb_s, axis=0, keepdims=True)) / np.std(emb_s, axis=0, keepdims=True)
            # m为多加的一行
            #emb_m = (emb_m - np.mean(emb_m, axis=0, keepdims=True) / np.std(emb_m, axis=0, keepdims=True))
            # Saving embeddings
            #pickle.dump((emb_c, emb_t, emb_s), open('../emb/emb_dblp_seed_{}_dim_{}'.format(seed, d), 'wb'))
            pickle.dump((emb_c, emb_t),open('../emb/emb_dblp_seed_{}_dim_{}'.format(seed, d), 'wb'))



#用于wd的数据库，里面的每个embed函数为中文的版本
def embed_wd(pypinyin=None):
    import jieba,zhconv,re
    from pypinyin import lazy_pinyin
    from gensim.models.word2vec import LineSentence
    from gensim.corpora import WikiCorpus

    def tokenizer_cn(text, token_min_len=10, token_max_len=100, lower=False):
        text = zhconv.convert(text,'zh-hans').strip() #Standardize to simple Chinese
        text = p.sub('',text)
        return jieba.lcut(text)

    def preproc_cn(docs,min_len=2,max_len=15):
        docs = [tokenizer_cn(doc) for doc in docs]
        # Removing Stop words
        stop_words = pickle.load(open('../data/wd/stop_words_cn.pkl','rb'))
        stop_words = set(stop_words)
        docs = [[word for word in document if word not in stop_words] for document in docs]
        return docs

    def clean_text_cn(text, lower=False):
        # 字符简化：将繁体中文转换为简体中文
        text = zhconv.convert(text, 'zh-hans').strip()

        # 文本清洗：使用正则表达式 p 移除特定字符或模式
        # 请确保 p 已经定义，比如 p = re.compile('[一些要移除的模式]')
        text = p.sub('', text)

        # 是否转换为小写
        if lower:
            text = text.lower()

        return text

    def process_wiki(inp, outp, dct):
        _ = dct[0]
        output = open(outp, 'w', encoding='utf-8')
        wiki = WikiCorpus(inp, processes=os.cpu_count()-2,
                          dictionary=dct, article_min_tokens=10,
                          lower=False)  # It takes about 16 minutes by 10 core cpu.
        count=0
        for words in wiki.get_texts():
            words = [" ".join(tokenizer_cn(w)) for w in words]
            output.write(' '.join(words) + '\n')
            count+=1
            if count%10000==0:
                print('Finished %d-67'%count//10000)
        output.close()

    def topic_embed_cn(docs,dim=100):
        return my_embed_cn(docs,dim)

    def char_embed_cn(docs,dim=16):
        docs = [''.join(lazy_pinyin(doc)).lower() for doc in docs]
        return char_embed(docs,dim=dim)

    p = re.compile('[^\u4e00-\u9fa5]')
    ex_corpus_xml = '../data/wd/zhwiki-latest-pages-articles.xml.bz2'
    ex_corpus_fname = '../data/wd/zhwiki_corpus'
    
    print(time.ctime(),'\tLoading data...')
    g1,g2 = pickle.load(open('../data/wd/networks','rb'))
    print(g1)
    print(time.ctime(),'\t Size of two networks:',len(g1),len(g2))    
    attrs = pickle.load(open('../data/wd/attrs','rb'))
    char_corpus,word_corpus,topic_corpus = [],[],[]    
    for i in range(len(attrs)):
        v = attrs[i]
        char_corpus.append(v[0]) 
        #word_corpus.append(v[1])
        topic_corpus.append(v[2])
        # The index number is the node id of users in the two networks.

    print(time.ctime(),'\tPreprocessing...')
    #word_corpus = preproc_cn(word_corpus)
    #topic_corpus = preproc_cn(topic_corpus)
    for seed in range(3):
        for d in [100]:
            print(time.ctime(),'\tCharacter level attributes embedding...')
            emb_c = char_embed_cn(char_corpus,dim=d)

            print(time.ctime(),'\tTopic level attributes embedding...')
            emb_t = topic_embed_cn(topic_corpus,dim=d)
            
            print(time.ctime(),'\tNetwork 1-1 embedding...')
            emb_g1 = network_embed(g1,method='line',dim=d)
            
            print(time.ctime(),'\tNetwork 1-2 embedding...')
            emb_g2 = network_embed(g2,method='line',dim=d)
            emb_g1.update(emb_g2)
            emb_s = np.array([emb_g1[str(i)] for i in range(len(emb_g1))])

            # Standardization
            emb_c = (emb_c-np.mean(emb_c,axis=0,keepdims=True))/np.std(emb_c,axis=0,keepdims=True)
            #emb_w = (emb_w-np.mean(emb_w,axis=0,keepdims=True))/np.std(emb_w,axis=0,keepdims=True)
            emb_t = (emb_t-np.mean(emb_t,axis=0,keepdims=True))/np.std(emb_t,axis=0,keepdims=True)
            emb_s = (emb_s-np.mean(emb_s,axis=0,keepdims=True))/np.std(emb_s,axis=0,keepdims=True)

            # Saving embeddings
            pickle.dump((emb_t,emb_s),open('../emb/emb_wd_seed_{}_dim_{}'.format(seed,d),'wb'))
            # pickle.dump((emb_c,emb_t,emb_s),open('../emb/emb_wd_seed_{}_dim_{}'.format(seed,d),'wb'))

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
#inputs = input('Selecte the dataset(0/1): (0为dblp数据库，可用，1为微博数据库中文的)')
inputs = 1
if int(inputs) == 0:
    print('Embedding dataset: dblp')
    embed_dblp()
    # 可以用这个，英文的科研数据库
else:
    print('Embedding dataset: wd')
    # 带中文的应该是微博数据库
    embed_wd()
