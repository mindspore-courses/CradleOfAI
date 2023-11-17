"""
author: Ruben Tao
"""

import dataset as d
import vocab as v
from model import BasicModel, BasicLoss, BasicEvaluation, BasicF1
import numpy as np
import mindspore
from mindspore import nn, Tensor
from mindspore.train.callback import LossMonitor, TimeMonitor, SummaryCollector

# 训练
def train():
    # 获得字表、字典
    vocab, dictionary = v.get_vocab_and_dictionary()
    # 获得训练数据集（句子的字索引+句子的字标签索引）
    dataset = d.get_dataset('dev.txt', dictionary).batch(256, drop_remainder=True)
    # 建立LSTM神经网络
    basic_model = BasicModel(vocab_size=len(vocab), embedding_dim=30, hidden_size=40, hidden_num_layer=2, label_size=12)

    # 尝试寻找模型参数文件，若存在，则直接加载模型
    try:
        param_dict = mindspore.load_checkpoint('model.ckpt')
        mindspore.load_param_into_net(basic_model, param_dict)
    except ValueError:
        print('未找到模型参数文件，将使用随机初始化参数')
        pass
    # 损失函数
    loss_fn = BasicLoss()
    # 优化器
    optimizer = nn.Adam(basic_model.trainable_params(), learning_rate=0.01)
    # 设置批次大小
    basic_model.set_batch_size(256)
    # 建立完整模型
    model = mindspore.Model(network=basic_model, loss_fn=loss_fn, optimizer=optimizer)
    # 训练
    model.train(epoch=10, train_dataset=dataset, callbacks=[LossMonitor(), TimeMonitor(),
                                                            SummaryCollector(summary_dir='/root/summary/baseline1', collect_freq=10)], dataset_sink_mode=False)
    # 保存参数
    mindspore.save_checkpoint(basic_model, 'model.ckpt')
    print('训练完成，模型已保存')

# 计算f1值
def evaluate_f1():
    vocab, dictionary = v.get_vocab_and_dictionary()
    # 获得测试数据集
    dataset = d.get_dataset('test.txt', dictionary).source
    basic_model = BasicModel(vocab_size=len(vocab), embedding_dim=30, hidden_size=40, hidden_num_layer=2, label_size=12)
    # 加载训练好的模型
    try:
        param_dict = mindspore.load_checkpoint('model.ckpt')
        mindspore.load_param_into_net(basic_model, param_dict)
    except ValueError:
        pass
    length = len(dataset)
    basic_model.set_batch_size(length)
    f1_net = BasicF1()
    inputs, targets = dataset[0:length] # 获得inputs(句子)、targets(答案标签)
    inputs = Tensor(np.array(inputs))   # 将inputs转换成张量
    targets = Tensor(np.array(targets)) # 将targets转换成张量
    outputs = basic_model(inputs)       # 通过模型获得预测标签
    f1_net.clear()
    f1_net.update(outputs, targets)
    f1 = f1_net.eval()  # 计算f1值
    print('f1: ', f1)


def evaluate_acc():
    vocab, dictionary = v.get_vocab_and_dictionary()
    basic_model = BasicModel(vocab_size=len(vocab), embedding_dim=30, hidden_size=40, hidden_num_layer=2, label_size=12)
    # 加载训练好的模型
    try:
        param_dict = mindspore.load_checkpoint('model.ckpt')
        mindspore.load_param_into_net(basic_model, param_dict)
    except ValueError:
        pass
    inputs, targets = d.get_dataset('test.txt', dictionary, padding=False)  # 获得inputs(句子)、targets(答案标签)
    length = len(inputs)
    basic_model.set_batch_size(1)
    eval_net = BasicEvaluation()
    i = 0
    total_acc = 0
    # 计算ACC
    while i < length:
        outputs = basic_model(Tensor(np.array(inputs[i:i + 1])))
        eval_net.clear()
        eval_net.update(outputs, Tensor(np.array(targets[i:i + 1])))
        acc = eval_net.eval()
        print(i, acc)
        total_acc += acc
        i += 1
    print('acc: ', total_acc / i)


if __name__ == '__main__':
    train()
    evaluate_acc()
    evaluate_f1()
