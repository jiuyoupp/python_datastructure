import random
import math
# Third-party libraries
import numpy as np
from numpy import *


class Homework(object):
    # 网络初始化
    def __init__(self, key):
        self.num_layers = len(key)  # 各层神经元个数 784 30 10
        self.key = key
        self.biases1 = np.random.randn(key[1], 1)   # 偏置初始化 [30x1,10x1]
        self.biases2 = np.random.randn(key[2], 1) 
        self.weight1 = np.random.randn(key[1],key[0])
        self.weight2 = np.random.randn(key[2], key[1])
        self.outy = np.random.randn(10, 1) 
    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        h = sigmoid(np.dot(self.weight1, a) + self.biases1)
        a = sigmoid(np.dot(self.weight2, h) + self.biases2)

        return a
                        
    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):
       
        training_data = list(training_data)  # 50000个样本
        n = len(training_data)

        if test_data:
            test_data = list(test_data)  # 10000个样本
            n_test = len(test_data)

        for j in range(epochs):
            random.shuffle(training_data)   #将元素随机排序
            mini_batches = [
                training_data[k:k + mini_batch_size]
                for k in range(0, n, mini_batch_size)] # 将50000个样本分成50000/mini_batch_size批分别运行
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta) # 一批一批运行每批有10个测试数据
            if test_data:
                print("训练次数{} : {} / {}".format(j, self.evaluate(test_data), n_test))
            else:
                print("训练结束 {} complete".format(j))
    def update_mini_batch(self, mini_batch, eta):

         nabla_b1 = [np.zeros(b1.shape) for b1 in self.biases1]
         nabla_w1 = [np.zeros(w1.shape) for w1 in self.weight1]    #生产b，w中相应神经层矩阵大小的0向量。其中w、b为各层偏置矩阵组合而成
         nabla_b2 = [np.zeros(b2.shape) for b2 in self.biases2]
         nabla_w2 = [np.zeros(w2.shape) for w2 in self.weight2]
         
         for x, y in mini_batch:  # 还是一个一个样本进行反向传播计算  mini_batch中含有mini_batch_size个样本 每个样本都是traindata其中含有输入x和输出标签y
             delta_nabla_b2, delta_nabla_w2,delta_nabla_b1, delta_nabla_w1 = self.backprop(x, y)
             nabla_b1 = [nb + dnb for nb, dnb in zip(nabla_b1, delta_nabla_b1)]
             nabla_w1 = [nw + dnw for nw, dnw in zip(nabla_w1, delta_nabla_w1)]
             nabla_b2 = [nb + dnb for nb, dnb in zip(nabla_b2, delta_nabla_b2)]
             nabla_w2 = [nw + dnw for nw, dnw in zip(nabla_w2, delta_nabla_w2)]
         self.weight1 = [w - (eta / len(mini_batch)) * nw          # 只不过更新权重，一批只更新一次
                         for w, nw in zip(self.weight1, nabla_w1)]
         self.biases1 = [b - (eta / len(mini_batch)) * nb
                         for b, nb in zip(self.biases1, nabla_b1)]
         self.weight2 = [w - (eta / len(mini_batch)) * nw          # 只不过更新权重，一批只更新一次
                         for w, nw in zip(self.weight2, nabla_w2)]
         self.biases2 = [b - (eta / len(mini_batch)) * nb
                         for b, nb in zip(self.biases2, nabla_b2)]
     # 反向传播
    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b1 = [np.zeros(b1.shape) for b1 in self.biases1]
        nabla_w1 = [np.zeros(w1.shape) for w1 in self.weight1]    #生产b，w中相应神经层矩阵大小的0向量。其中w、b为各层偏置矩阵组合而成
        nabla_b2 = [np.zeros(b2.shape) for b2 in self.biases2]
        nabla_w2 = [np.zeros(w2.shape) for w2 in self.weight2]
        # feedforward
        activation = x             # 存储每层的z和a值
        activations = [x]  # list to store all the activations, layer by layer
        zs = []  # list to store all the z vectors, layer by layer
        z = np.dot(self.weight1, activation) + self.biases1
        zs.append(z)
        activation = sigmoid(z)
        activations.append(activation)

        z = np.dot(self.weight2, activation) + self.biases2
        zs.append(z)
        activation = sigmoid(z)
        activations.append(activation)


        # backward pass 计算最后一层的误差
        delta = self.cost_derivative(activations[-1], y) * \
                sigmoid_prime(zs[-1])
        nabla_b1 = delta
        nabla_w1 = np.dot(delta, activations[-2].transpose())
        # 计算从倒数第二层至第二层的误差
        for l in range(2, self.num_layers): 
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(np.transpose(self.weight2), delta) * sp   ##
            nabla_b2 = delta
            nabla_w2 = np.dot(delta, activations[-l - 1].transpose())  ##deltaw=步长*deta*上一层的输出

        return (nabla_b1, nabla_w1, nabla_b2, nabla_w2)

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations - y)

   
# Miscellaneous functions
def sigmoid(z):
    """The sigmoid function."""
    return 1.0 / (1.0 + np.exp(-z))


# 导数
def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z) * (1 - sigmoid(z))
    
        

