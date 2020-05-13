# -*- coding:utf-8 -*-
#! usr/bin/env python3
"""
Created on 10/04/2020 上午12:33 
@Author: xinzhi yao
"""
import io
import collections
import string
import os
import re
import time
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
from collections import Counter, defaultdict
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Step 1: Data preprocessing.
# 对文本进行标准化
# 其中除了'-'的标点替换为空格 数字替换为NBR 文本全部替换为小写 去除停用词
def str_norm(str_list: list, punc2=' ', num2='NBR', space2=' ', lower=True):
    punctuation = string.punctuation.replace('-', '')
    rep_list = str_list.copy()
    for index, row in enumerate(rep_list):
        row = row.strip()
        row = re.sub("\d+.\d+", num2, row)
        row = re.sub('\d+', num2, row)
        for pun in punctuation:
            row = row.replace(pun, punc2)
        if lower:
            row = row.lower()
        rep_list[index] = re.sub(' +', space2, row)
    return rep_list

def Data_Pre(corpus: str, out: str, head = True):
    # wnl = WordNetLemmatizer()
    # if os.path.exists((out)):
        # return out
    wf = open(out, 'w')
    with open(corpus) as f:
        if head:
            l = f.readline()
        for line in f:
            l = line.strip()
            sent_list = str_norm([l], punc2=' ', num2='NBR', space2=' ')
            for sent in sent_list:
                wf.write('{0}\n'.format(sent))
    wf.close()
    return out

print('-'*50)
raw_file = './All_table.txt'

corpus = Data_Pre(raw_file, './corpus.txt')
stopwords = stopwords.words('english')
# Read the data into a list of strings.
def read_data(filename: str):
    words = []
    with open(filename) as f:
        for line in f:
            l = line.strip().split()
            for word in l:
                if word not in stopwords:
                    words.append(word)       
    return words

words = read_data((corpus))
print('Data size: {0}.'.format(len(words)))

# Step 2: Build the dictionary and replace rare words with UNK token
# 将data转化为index, 同时将单词按词频排序 保留前 vocabulary_size 个单词 其他单词替换为 'UNK'
# 如果负采样包含太多低频单词 学到的嵌入太集中
def build_dataset(words, low_fre: 5):
    # count = [['UNK', -1]]
    # count.extend(collections.Counter(words))
    dictionary = dict()

    count_dic = defaultdict(int)
    count_dic['UNK'] = -1
    for i in words:
        count_dic[i] += 1

    for word, fre in count_dic.items():
        if fre > low_fre:
            dictionary[word] = len(dictionary)
    data = []
    unk_count = 0
    word_set = set(dictionary.keys())
    for word in words:
        if word in word_set:
            index = dictionary[word]
        else:
            index = 0
            unk_count += 1
        data.append(index)
    # count[0][1] = unk_count
    count_dic['UNK'] = unk_count

    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, dictionary, reversed_dictionary

data, dictionary, reverse_dictionary = build_dataset(words, low_fre=5)
words = list(dictionary.keys())
# words = [reverse_dictionary[i] for i in data]
print('Vocabulary size: {0}.'.format(len(words)))
# print('Most common words (+UNK)', count[:5])
print('Sample data', data[:10], [reverse_dictionary[i] for i in data[:10]])

# Step 3: Function to generate a training batch for the skip-gram model.
# 生成 batch_data 根据 num_skips 和 skip_window 返回 中心词index 和 上下文单词index
def generate_batch(data, batch_size, num_skips, skip_window):
  global data_index
  assert batch_size % num_skips == 0
  assert num_skips <= 2 * skip_window
  batch = np.ndarray(shape=(batch_size), dtype=np.int32)
  labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
  span = 2 * skip_window + 1 # [ skip_window target skip_window ]
  buffer = collections.deque(maxlen=span)
  for _ in range(span):
    buffer.append(data[data_index])
    data_index = (data_index + 1) % len(data)
  for i in range(batch_size // num_skips):
    target = skip_window  # target label at the center of the buffer
    targets_to_avoid = [ skip_window ]
    for j in range(num_skips):
      while target in targets_to_avoid:
        target = random.randint(0, span - 1)
      targets_to_avoid.append(target)
      batch[i * num_skips + j] = buffer[skip_window]
      labels[i * num_skips + j, 0] = buffer[target]
    buffer.append(data[data_index])
    data_index = (data_index + 1) % len(data)
  return batch, labels

data_index = 0

print('batch data examples: ')
batch, labels = generate_batch(data=data, batch_size=8, num_skips=2, skip_window=1)
for i in range(8):
  print(batch[i], reverse_dictionary[batch[i]],
      '->', labels[i, 0], reverse_dictionary[labels[i, 0]])


# Step 4: Build a skip-gram model.
# 使用负采样的 Skip-Gram 模型定义
class SkipGramNeg(nn.Module):
    def __init__(self, args):
        super(SkipGramNeg, self).__init__()

        # 是否使用 GPU
        self.use_cuda = args.use_cuda
        # V
        self.vocabulary_size = args.vocabulary_size
        # d
        self.embedding_size = args.embedding_size
        # Noise data distribution (Word frequency)
        self.noise_dist = args.noise_dist

        # define embedding layers for input and output words
        # W
        self.in_embed = nn.Embedding(self.vocabulary_size, self.embedding_size)
        # W'
        self.out_embed = nn.Embedding(self.vocabulary_size, self.embedding_size)

    def weight_init(self):
        # 权重初始化
        self.in_embed.weight.data.uniform_(-1, 1)
        self.out_embed.weight.data.uniform_(-1, 1)

    def similarity_eval(self, embedding, valid_data):
        # 计算验证集embedding和全部embedding的余弦相似度
        embedding = F.normalize(embedding)
        valid_embeddings = self.forward_output(valid_examples)
        valid_embeddings = F.normalize(valid_embeddings)
        similarity = valid_embeddings.mm(embedding.t())
        return similarity

    def forward_input(self, input_words):
        # input lookup
        input_vectors = self.in_embed(input_words)
        return input_vectors

    def forward_output(self, output_words):
        # output lookup
        output_vectors = self.out_embed(output_words)
        return output_vectors

    def forward_noise(self, batch_size, n_samples):
        """ Generate noise vectors with shape (batch_size, n_samples, n_embed)"""
        # Noise data distribution
        if self.noise_dist is None:
            noise_dist = torch.ones(self.vocabulary_size)
            print(noise_dist.shape)
        else:
            noise_dist = self.noise_dist

        # Sample words from our noise distribution
        noise_words = torch.multinomial(noise_dist,
                                        batch_size * n_samples,
                                        replacement=True)

        # gpu
        if self.use_cuda:
            noise_words = noise_words.to('cuda')

        # 转换为embedding
        noise_vectors = self.out_embed(noise_words).view(batch_size, n_samples, self.embedding_size)

        return noise_vectors

# NCE Loss 定义
class NegativeSamplingLoss(nn.Module):
    def __init__(self):
        super(NegativeSamplingLoss, self).__init__()

    def forward(self, input_vectors, output_vectors, noise_vectors):
        batch_size, embed_size = input_vectors.shape

        # Input vectors should be a batch of column vectors
        input_vectors = input_vectors.view(batch_size, embed_size, 1)

        # Output vectors should be a batch of row vectors
        output_vectors = output_vectors.view(batch_size, 1, embed_size)

        # bmm = batch matrix multiplication
        # correct log-sigmoid loss
        out_loss = torch.bmm(output_vectors, input_vectors).sigmoid().log()
        out_loss = out_loss.squeeze()

        # incorrect log-sigmoid loss
        noise_loss = torch.bmm(noise_vectors.neg(), input_vectors).sigmoid().log()
        noise_loss = noise_loss.squeeze().sum(1)  # sum the losses over the sample of noise vectors

        # negate and sum correct and noisy log-sigmoid losses
        # return average batch loss
        return -(out_loss + noise_loss).mean()

# Step 5: Begin training.
# 超参数
class config():
    def __init__(self):
        self.epoch = 5
        self.num_steps = 1000
        self.batch_size = 512
        self.check_step = 1
        self.eval_step = 300

        self.vocabulary_size = 40000
        self.embedding_size = 128  # Dimension of the embedding vector.
        self.noise_dist = None
        self.skip_window = 5  # How many words to consider left and right.
        self.num_skips = 4  # How many times to reuse an input to generate a label.

        self.threshold = 1e-5
        self.SubSampling = False

        # We pick a random validation set to sample nearest neighbors. Here we limit the
        # validation samples to the words that have a low numeric ID, which by
        # construction are also the most frequent.
        self.valid_size = 16  # Random set of words to evaluate similarity on.
        self.valid_window = 100  # Only pick dev samples in the head of the distribution.
        self.num_sampled = 64  # Number of negative examples to sample.
        self.use_cuda = torch.cuda.is_available()

        self.lr = 0.025
        self.momentum = 0

args = config()
# fixme
args.vocabulary_size = len(words)
# 计算词频
word_counts = Counter(data)
total_count = len(data)
freqs = {word: count / total_count for word, count in word_counts.items()}

# Calculate noise data distribution
word_freqs = np.array(sorted(freqs.values(), reverse=True))
unigram_dist = word_freqs / word_freqs.sum()
noise_dist = torch.from_numpy(unigram_dist ** (0.75) / np.sum(unigram_dist ** (0.75)))
args.noise_dist = noise_dist

print('Model: ')
model = SkipGramNeg(args)
print(model)

if args.use_cuda:
    model = model.to('cuda')

# 定义损失函数 和 优化器
criterion = NegativeSamplingLoss()
optimizer = optim.Adam(model.parameters(), lr=args.lr)

# 随机选取 验证样本
valid_examples = np.random.choice(args.valid_window, args.valid_size, replace=False)
valid_examples = torch.LongTensor(valid_examples)
if args.use_cuda:
    valid_examples = valid_examples.to('cuda')

print('-'*50)
print('Start training.')
start_time = time.time()
average_loss = 0
average_losss = []
average_losss.append(average_loss)
for step in range(1, args.num_steps):
    batch_inputs, batch_labels = generate_batch(
        data, args.batch_size, args.num_skips, args.skip_window)
    batch_labels = batch_labels.squeeze()
    batch_inputs, batch_labels = torch.LongTensor(batch_inputs), torch.LongTensor(batch_labels)
    if args.use_cuda:
        batch_inputs, batch_labels = batch_inputs.to('cuda'), batch_labels.to('cuda')

    # input, output, and noise vectors
    input_vectors = model.forward_input(batch_inputs)
    output_vectors = model.forward_output(batch_labels)
    noise_vectors = model.forward_noise(batch_inputs.size(0), 5)

    # negative sampling loss
    loss = criterion(input_vectors, output_vectors, noise_vectors)
    average_loss += loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % args.check_step == 0:
        average_loss /= args.check_step
        end_time = time.time()
        # The average loss is an estimate of the loss over the last 2000 batches.
        print('Average loss as step {0}: {1:.2f}, cost: {2:.2f}s.'.format(step, average_loss, end_time-start_time))
        average_losss.append(average_loss.cpu().detach().numpy())
        average_loss = 0
        start_time = time.time()


    if step % args.eval_step == 0:
        print('-'*50)
        print('Evaluation.')
        model.eval()
        Sim = model.similarity_eval(model.in_embed.weight.data, valid_examples)
        for i in range(args.valid_size):
            valid_word = reverse_dictionary[int(valid_examples[i].data)]
            top_k = 8
            nearest = (-Sim[i, :]).argsort()[1: top_k+1]
            log_str = "Nearest to: '{0}' is ".format(valid_word)
            for k in range(top_k):
                close_word = reverse_dictionary[int(nearest[k])]
                log_str = "{0} {1},".format(log_str, close_word)
            print(log_str)
        model.train()
        print('-'*50)
average_losss = np.array(average_losss)
np.save("loss_lr0.0003.npy",average_losss)
print('Training complete.')
print('-'*50)
final_embedding = model.in_embed.weight.data.cpu().numpy()


# Step 6: Visualize the embeddings.
print('Embedding visualization.')

out_v = io.open('vecs.tsv', 'w', encoding='utf-8')
out_m = io.open('meta.tsv', 'w', encoding='utf-8')
for i in range(1000):
    label = reverse_dictionary[i]
    embedding = final_embedding[i]
    out_m.write(str(label) + "\n")
    out_v.write('\t'.join([str(x) for x in embedding]) + "\n")
    
out_v.close()
out_m.close()

dictionary = {v: k for k, v in reverse_dictionary.items()}

def cal_dis(file,dictionary,final_embedding):
    word_list = pd.read_csv(file)
    loss = 0.0
    word_len = len(word_list)
    print(word_len)
    for i in range(word_len):
        word_1 = final_embedding[dictionary[word_list.loc[i][0]]]
        word_2 = final_embedding[dictionary[word_list.loc[i][1]]]
        print(word_1)
        word_1 = np.mat(word_1)
        word_2 = np.mat(word_2)
        loss += np.linalg.norm(word_1-word_2)
    return 1/word_len * loss


print('TSNE visualization is completed')


