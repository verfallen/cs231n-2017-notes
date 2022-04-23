# 回顾

上节课我们讨论了不同的深度学习框架,讨论了 PyTorch，Tensor Flow还有 Caffe2。深度学习框架的优点有：

+ 快速构建大型的运算网络，例如大规模神经网络和卷积神经网络。
+ 快速计算网络中的梯度，同时能够计算所有中间变量的权重并用来训练模型。
+ 在 GPU 上高效运行

这些框架主要是通过调制神经网络中的前向层和后向层来工作。使用时只需要定义神经网络层的顺序。就可以很快构建一个很复杂的神经网络架构。

# 介绍

会讨论一些特定类型的卷积神经网络架构，在研究和实际应用中使用得很广泛。深入探讨那些 ImageNet大赛获胜者用的最多的神经网络架构，按照时间顺序它们分别是是 AlexNet，VGGNet，GoogleNet 和 ResNet。然后简单介绍些其他的目前并不常用的架构。

# LeNet

LeNet可以看作是卷积网络的第一个实例，并且在实际应用中取得成功。

下图就是 LeNet 的结构，输入一个图片，使用步长为1大小为5x5的卷积核，接下来重复卷积和池化操作，最后有一些全连接层。LeNet在数字识别领域的应用方面取得了成功

<img src="https://raw.githubusercontent.com/verfallen/cs231n-2017-notes/main/img/202204221352033.png" alt="image-20220422135217947" style="zoom:67%;" />

# AlexNet

AlexNet是第一个在 ImageNet 的分类比赛中获得成功的大型卷积神经网络。AlexNet 在2012年参赛，之前的非深度学习架构相比，它大幅提高了识别准确率。从此开始了大规模对卷积神经网络的研究和应用。

下图是**AlexNet 的基础架构**：卷积层，池化层，归一化层，卷积，池化，归一化层然后是三个卷积层一个池化层，最后是三个全连接层。它看上去和 LeNet 很相似，区别在于总层数变多了，卷积层达到了五层，在最后的全连接层输出分类之前，多了两层全连接层。

![image-20220422144714052](https://raw.githubusercontent.com/verfallen/cs231n-2017-notes/main/img/202204221447154.png)