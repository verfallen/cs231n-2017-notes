# 回顾

上节课我们讨论了深度学习的优化算法，包括SGD动量，也可以对SGD动量进行一点微调。也讨论了正则化，尤其要记住的是 dropout，它在正向传播过程中将网络的某些部分随机的设置为0，然后在测试时消除这些随机性。我们发现在不同的正则化中，这是普遍的模式：在深度学习的训练中加入一些噪声，在测试时再将噪声边缘化。我们还讨论了迁移学习，可以先下载一些相似的大的数据集预训练网络模型，然后根据自己的实际任务进行微调。这个方法可以帮助你在没有很大数据集的情况下也能完成深度学习任务。

# 前言

今天要讲一些关于编写软件和硬件，以及它们如何工作的细节。以及了解一下软件的实际应用流程，讨论一些主流的 CPU 和GPU。然后讨论几个主流的深度学习框架。

# CPU vs GPU

我们已经知道，计算机有CPU和GPU，深度学习使用GPU。但是并没有指出它们具体是什么，以及为什么在深度学习中GPU的表现要更好。

## CPU的介绍

CPU的全程是 Center Processing Unit ,中央处理器。就是下图这个小芯片。

<img src="https://raw.githubusercontent.com/verfallen/cs231n-2017-notes/main/img/image-20220416230736091.png" alt="image-20220416230736091" style="zoom:50%;" />

## GPU的介绍

CPU 的全程是 Graphic Processing Unit，图形处理器。相对于CPU来说，GPU的体积大得多，它有自己的冷却系统，并且耗电更多。

从GPU的名字就可以看出来，他最初是被用于渲染计算机图形，是围绕游戏而开发的。

<img src="https://raw.githubusercontent.com/verfallen/cs231n-2017-notes/main/img/image-20220416231035325.png" alt="image-20220416231035325" style="zoom: 50%;" />

我们知道的GPU一般有NVIDIA 和 AMD。深度学习中，我们基本上选择NVIDIA。使用AMD的显卡来做深度学习可能会遇到各种麻烦。

事实上，NVIDIA 一直大力推动深度学习。他们投入了很多人力物力来让他们的显卡更适合深度学习。所以，谈起深度学习中的GPU通常实质NVIDIA的GPU。

