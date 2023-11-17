"""
author: Ruben Tao
"""

from typing import List, Dict
import numpy as np
import tokenizer
from mindspore.dataset import GeneratorDataset
import vocab


class DatasetGenerator:
    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets

    def __getitem__(self, item):
        return self.inputs[item], self.targets[item]

    def __len__(self):
        return len(self.inputs)

# 加载数据集，返回句子、标签、最大长度
def _load_dataset(path: str):
    print('loading dataset: ' + path)
    sentences = []
    labels = []
    max_length = 0
    with open(path, mode='r') as fp:
        sentence = fp.readline()
        while sentence:
            label = fp.readline().rstrip()
            sentence = sentence.rstrip()
            if len(sentence) > max_length:
                max_length = len(sentence)
            sentences.append(sentence)
            labels.append(label)
            sentence = fp.readline()
    print('sentences size: ' + str(len(sentences)) + ', labels size: ' + str(len(labels)) + ', sentence max length: ' + str(max_length))
    return sentences, labels, max_length


def read_sentences_from_dataset(path_list: List[str]):
    sentences = []
    for path in path_list:
        with open(path, mode='r') as fp:
            sentence = fp.readline()
            while sentence:
                fp.readline()
                sentences.append(sentence.rstrip())
                sentence = fp.readline()
    return sentences


def _construct_dataset_generator(sentences: List[str], labels: List[str], max_length: int, dictionary: Dict):
    inputs = []
    # 根据字典，将句子转换成字的索引
    for sentence in sentences:
        inputs.append(tokenizer.tokenize(sentence, dictionary, max_length))
    targets = []
    # 根据内置字典，将标签转换成索引
    for label in labels:
        targets.append(tokenizer.tokenize(label.split(' '), length=max_length))
    if max_length == 0:
        return inputs, targets

    inputs = np.array(inputs, dtype=np.int32)
    targets = np.array(targets, dtype=np.int32)
    return DatasetGenerator(inputs, targets)


def get_dataset(dataset_path: str, dictionary: Dict, padding: bool = True):
    sentences, labels, max_length = _load_dataset(dataset_path)
    if not padding:
        max_length = 0
        return _construct_dataset_generator(sentences, labels, max_length, dictionary)
    dataset_generator = _construct_dataset_generator(sentences, labels, max_length, dictionary)
    print('dataset generator size: ' + str(len(dataset_generator)))
    return GeneratorDataset(dataset_generator, column_names=['inputs', 'labels'])


def _test():
    _, dictionary = vocab.get_vocab_and_dictionary()
    dataset = get_dataset('dev.txt', dictionary)
    print(dataset.get_dataset_size())


if __name__ == '__main__':
    _test()
