"""
author: Ruben Tao
"""

from mindspore import nn, Tensor
import numpy as np
import mindspore
from mindspore.common.initializer import initializer


class BasicModel(nn.Cell):
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_size: int,
                 hidden_num_layer: int, label_size: int, batch_size: int = 64):
        # vocab_size 词表大小
        # embedding_dim 字向量维度
        # hidden_size 隐含层向量维度
        # hidden_num_layer 隐含层层数
        # label_size 标签类别数
        super(BasicModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size=vocab_size, embedding_size=embedding_dim)  # 将字索引转换成字向量
        self.hidden_size = hidden_size
        # 将隐含层结构设置为长短时记忆网络
        # cuDNN（基于CUDA的深度学习GPU加速库）中RNN的API的batch_size在第二维度，便于并行计算
        # 但是习惯上，我们会将batch_size设为第一维度，即设置batch_first=True
        self.hidden = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size, num_layers=hidden_num_layer,
                              batch_first=True, bidirectional=True)
        # 初始化隐状态
        self.h0 = initializer('zeros', [2 * hidden_num_layer, batch_size, hidden_size], dtype=mindspore.float32)
        # 初始化记忆细胞
        self.c0 = initializer('zeros', [2 * hidden_num_layer, batch_size, hidden_size], dtype=mindspore.float32)
        # 设置全连接层
        self.linear = nn.Dense(in_channels=2 * hidden_size, out_channels=label_size)
        self.isTraining = True

    def construct(self, inputs: mindspore.Tensor):
        outputs = self.embedding(inputs)
        outputs, _ = self.hidden(outputs, (self.h0, self.c0))
        outputs = self.linear(outputs)
        return outputs


    def set_batch_size(self, batch_size):
        self.h0 = initializer('zeros', [2 * self.hidden.num_layers, batch_size, self.hidden_size], dtype=mindspore.float32)
        self.c0 = initializer('zeros', [2 * self.hidden.num_layers, batch_size, self.hidden_size], dtype=mindspore.float32)


class BasicLoss(nn.loss.LossBase):
    def __init__(self):
        super(BasicLoss, self).__init__()
        self.loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='sum')

    def construct(self, logits, labels):
        return self.loss(logits.reshape((logits.shape[0] * logits.shape[1], logits.shape[2])), labels.reshape(labels.shape[0] * labels.shape[1]))


class BasicEvaluation(nn.Cell):
    def __init__(self):
        super(BasicEvaluation, self).__init__()
        self.onehot = nn.OneHot(depth=12)
        self.acc = nn.Accuracy(eval_type='multilabel')

    def clear(self):
        self.acc.clear()

    def update(self, *inputs):
        y_pred = inputs[0].argmax(axis=-1)
        y_pred = y_pred.reshape(y_pred.shape[0] * y_pred.shape[1])
        y_pred = self.onehot(y_pred)
        y = inputs[1]
        y = y.reshape(y.shape[0] * y.shape[1])
        y = self.onehot(y)
        self.acc.update(y_pred, y)

    def eval(self):
        return self.acc.eval()


class BasicF1(nn.Cell):
    def __init__(self):
        super(BasicF1, self).__init__()
        self.softmax = nn.Softmax()
        self.f1 = nn.F1()

    def clear(self):
        self.f1.clear()

    def update(self, *inputs):
        y_pred = self.softmax(inputs[0])
        y_pred = y_pred.reshape(y_pred.shape[0] * y_pred.shape[1], y_pred.shape[2])
        y = inputs[1]
        y = y.reshape(y.shape[0] * y.shape[1])
        self.f1.update(y_pred, y)

    def eval(self):
        return self.f1.eval()