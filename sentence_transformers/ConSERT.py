import random
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import batch_to_device


class ConSERT(SentenceTransformer):
    #模型所在位置
    def __init__(self, model_name_or_path=None, modules=None, device=None, cache_folder=None, cutoff_rate=0.15,
                 close_dropout=False):

        SentenceTransformer.__init__(self, model_name_or_path, modules, device, cache_folder)
        #print(type(SentenceTransformer))
        self.cutoff_rate = cutoff_rate
        self.close_dropout = close_dropout
        self.dup_rate = 0.32

    def smart_batching_collate(self, batch):
        """
        Transforms a batch from a SmartBatchingDataset to a batch of tensors for the model
        Here, batch is a list of tuples: [(tokens, label), ...]

        :param batch:
            a batch from a SmartBatchingDataset
        :return:
            a batch of tensors for the model
        """
        num_texts = len(batch[0].texts)
        texts = [[] for _ in range(num_texts)]
        labels = []

        for example in batch:
            for idx, text in enumerate(example.texts):
                texts[idx].append(text)

            labels.append(example.label)

        labels = torch.tensor(labels).to(self._target_device)

        sentence_features = []
        for idx in range(num_texts):
            tokenized = self.tokenize(texts[idx])
            batch_to_device(tokenized, self._target_device)
            sentence_features.append(tokenized)

        sentence_features[1] = self.shuffle_and_cutoff(sentence_features[1])
        sentence_features[1] = self.word_repetition(sentence_features[1])
        return sentence_features, labels

    def data_augment(self, tokenized):
        sentence_feature = tokenized
        batch_to_device(sentence_feature, self._target_device)
        sentence_feature = self.shuffle_and_cutoff(sentence_feature)
        sentence_features = self.word_repetition(sentence_feature)
        return sentence_features

    def shuffle_and_cutoff(self, sentence_feature):
        input_ids, attention_mask = sentence_feature['input_ids'], sentence_feature['attention_mask']
        bsz, seq_len = input_ids.shape
        shuffled_input_ids = []
        cutoff_attention_mask = []
        for bsz_id in range(bsz):
            sample_mask = attention_mask[bsz_id]
            num_tokens = sample_mask.sum().int().item()
            cur_input_ids = input_ids[bsz_id]
            if 102 not in cur_input_ids:  # tip:
                indexes = list(range(num_tokens))[1:]
                random.shuffle(indexes)
                indexes = [0] + indexes  # 保证第一个位置是0
            else:
                indexes = list(range(num_tokens))[1:-1]
                random.shuffle(indexes)
                indexes = [0] + indexes + [num_tokens - 1]  # 保证第一个位置是0，最后一个位置是SEP不变
            rest_indexes = list(range(num_tokens, seq_len))
            total_indexes = indexes + rest_indexes
            shuffled_input_id = input_ids[bsz_id][total_indexes]
            # print(shuffled_input_id,indexes)
            if self.cutoff_rate > 0.0:
                sample_len = max(int(num_tokens * (1 - self.cutoff_rate)),
                                 1)  # if true_len is 32, cutoff_rate is 0.15 then sample_len is 27
                start_id = np.random.randint(1,
                                             high=num_tokens - sample_len + 1)  # start_id random select from (0,6)，避免删除CLS
                cutoff_mask = [1] * seq_len
                for idx in range(start_id, start_id + sample_len):
                    cutoff_mask[idx] = 0  # 这些位置是0，bool之后就变成了False，而masked_fill是选择True的位置替换为value的
                cutoff_mask[0] = 0  # 避免CLS被替换
                cutoff_mask[num_tokens - 1] = 0  # 避免SEP被替换
                cutoff_mask = torch.ByteTensor(cutoff_mask).bool().to(input_ids.device)
                shuffled_input_id = shuffled_input_id.masked_fill(cutoff_mask, value=0).to(input_ids.device)
                sample_mask = sample_mask.masked_fill(cutoff_mask, value=0).to(input_ids.device)

            shuffled_input_ids.append(shuffled_input_id)
            cutoff_attention_mask.append(sample_mask)
        shuffled_input_ids = torch.vstack(shuffled_input_ids)
        cutoff_attention_mask = torch.vstack(cutoff_attention_mask)
        return {"input_ids": shuffled_input_ids, "attention_mask": cutoff_attention_mask, "token_type_ids": sentence_feature["token_type_ids"]}

    def word_repetition(self, sentence_feature):
        input_ids, attention_mask, token_type_ids = sentence_feature['input_ids'].cpu().tolist(
        ), sentence_feature['attention_mask'].cpu().tolist(), sentence_feature['token_type_ids'].cpu().tolist()
        bsz, seq_len = len(input_ids), len(input_ids[0])
        # print(bsz,seq_len)
        repetitied_input_ids = []
        repetitied_attention_mask = []
        repetitied_token_type_ids = []
        rep_seq_len = seq_len
        for bsz_id in range(bsz):
            sample_mask = attention_mask[bsz_id]
            actual_len = sum(sample_mask)

            cur_input_id = input_ids[bsz_id]
            # 在计算 dup_len 之前添加一个检查
            if actual_len <= 2:
                 #如果 actual_len 小于 2，将 dup_len 设置为 0 或者其他合适的值
                dup_len = random.randint(a=0, b=int(self.dup_rate * actual_len))
            else:
                dup_len = random.randint(a=0, b=max(2, int(self.dup_rate * actual_len)))
            '''
            # 确保 dup_len 处于有效范围内
            if dup_len < 0:
                print("错误：dup_len 为负数。")
            elif dup_len > actual_len:
                print("错误：dup_len 大于 actual_len。")
            else:
                # 生成一个要从中抽样的索引列表
                indices_to_sample = list(range(1, actual_len))

                if dup_len > 0:
                    # 检查 dup_len 是否大于可用索引的数量
                    if dup_len > len(indices_to_sample):
                        print("错误：dup_len 大于可用索引的数量。")
                        print(actual_len)
                        print(dup_len)
                        print(len(indices_to_sample))
                    else:
                        # 使用 random.sample() 抽样元素并打印结果
                        dup_word_index = random.sample(indices_to_sample, k=dup_len)
                        print("抽样成功。dup_word_index：", dup_word_index)
                else:
                    # 处理 dup_len 为 0 的情况
                    print("无需抽样。dup_word_index 是一个空列表。")
            '''
            dup_word_index = random.sample(
                list(range(1, actual_len)), k=dup_len)

            r_input_id = []
            r_attention_mask = []
            r_token_type_ids = []
            for index, word_id in enumerate(cur_input_id):
                if index in dup_word_index:
                    r_input_id.append(word_id)
                    r_attention_mask.append(sample_mask[index])
                    r_token_type_ids.append(token_type_ids[bsz_id][index])

                r_input_id.append(word_id)
                r_attention_mask.append(sample_mask[index])
                r_token_type_ids.append(token_type_ids[bsz_id][index])

            after_dup_len = len(r_input_id)
            # assert after_dup_len==actual_len+dup_len
            repetitied_input_ids.append(r_input_id)  # +rest_input_ids)
            repetitied_attention_mask.append(
                r_attention_mask)  # +rest_attention_mask)
            repetitied_token_type_ids.append(
                r_token_type_ids)  # +rest_token_type_ids)

            assert after_dup_len == dup_len + seq_len
            if after_dup_len > rep_seq_len:
                rep_seq_len = after_dup_len

        for i in range(bsz):
            after_dup_len = len(repetitied_input_ids[i])
            pad_len = rep_seq_len - after_dup_len
            repetitied_input_ids[i] += [0] * pad_len
            repetitied_attention_mask[i] += [0] * pad_len
            repetitied_token_type_ids[i] += [0] * pad_len

        repetitied_input_ids = torch.LongTensor(repetitied_input_ids).to(self.device)
        repetitied_attention_mask = torch.LongTensor(repetitied_attention_mask).to(self.device)
        repetitied_token_type_ids = torch.LongTensor(repetitied_token_type_ids).to(self.device)
        return {"input_ids": repetitied_input_ids, 'attention_mask': repetitied_attention_mask,
                'token_type_ids': repetitied_token_type_ids}


if __name__ == '__main__':
    #model = ConSERT()
    model = ConSERT('A:/bert-base-uncased', "cuda:0", cutoff_rate=0.15, close_dropout=True)
    model.__setattr__("max_seq_length", 64)
    print(model)