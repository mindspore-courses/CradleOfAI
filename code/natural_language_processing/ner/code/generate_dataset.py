"""
author: Ruben Tao
"""

import json


def main():
    with open('original_dataset.json', mode='r') as fp:
        source = json.load(fp)
    sentences = []
    labels = []
    examples = source.values()
    for example in examples:
        dialogue = example['dialogue']
        for element in dialogue:
            sentences.append(element['sentence'])
            labels.append(element['BIO_label'])
    print('sentences length: ' + str(len(sentences)) + ', labels length: ' + str(len(labels)))
    i = 0
    length = len(sentences)
    limit = int(0.8 * length)
    with open('train.txt', mode='w') as fp:
        while i < limit:
            fp.write(sentences[i] + '\n')
            fp.write(labels[i])
            if i < limit - 1:
                fp.write('\n')
            i += 1
    train_size = i
    print('train set size: ' + str(train_size))
    limit = int(0.9 * length)
    with open('dev.txt', mode='w') as fp:
        while i < limit:
            fp.write(sentences[i] + '\n')
            fp.write(labels[i])
            if i < limit - 1:
                fp.write('\n')
            i += 1
    dev_size = i - train_size
    print('dev set size: ' + str(dev_size))
    with open('test.txt', mode='w') as fp:
        while i < length:
            fp.write(sentences[i] + '\n')
            fp.write(labels[i])
            if i < length - 1:
                fp.write('\n')
            i += 1
    print('test set size: ' + str(length - train_size - dev_size))


if __name__ == '__main__':
    main()
