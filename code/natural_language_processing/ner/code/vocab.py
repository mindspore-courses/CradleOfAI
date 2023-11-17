"""
author: Ruben Tao
"""

from typing import List
import dataset


UNK_TOKEN = '<unk>'
PAD_TOKEN = '<pad>'

def construct_vocab_from_sentences(sentences: List[str], vocab_saving_path: str = '.'):
    vocab = set()
    for sentence in sentences:
        for ch in sentence:
            vocab.add(ch)
    vocab = list(vocab)
    with open(vocab_saving_path + '/vocab.txt', mode='w') as fp:
        for i in range(0, len(vocab)):
            fp.write(vocab[i])
            if i < len(vocab) - 1:
                fp.write('\n')

# 创建字表与字典
def get_vocab_and_dictionary(vocab_saving_path: str = 'vocab.txt'):
    vocab = []
    # 从vocab_saving_path文件中读字，创建字表（通过索引查找字）
    with open(vocab_saving_path, mode='r') as fp:
        temp = fp.readline()
        while temp:
            vocab.append(temp.rstrip())
            temp = fp.readline()
    dictionary = {}
    i = 0
    # 创建字典（字与索引的键值对，通过字查找索引）
    for ch in vocab:
        dictionary[ch] = i
        i += 1
    dictionary[UNK_TOKEN] = i
    vocab.append(UNK_TOKEN)
    i += 1
    dictionary[PAD_TOKEN] = i
    vocab.append(PAD_TOKEN)
    print('vocab size: ' +  str(len(vocab)) + ', dict size: ' + str(len(dictionary)))
    return vocab, dictionary


if __name__ == '__main__':
    construct_vocab_from_sentences(dataset.read_sentences_from_dataset(['train.txt', 'dev.txt', 'test.txt']))
    get_vocab_and_dictionary()