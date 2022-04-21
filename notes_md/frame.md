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

## CPU和GPU对比

CPU和GPU都是一种通用的计算机器。但是在性质上极为不同。

CPU的核数很少，可以利用多线程技术。也就是说，他们在硬件上可以同时运行多个线程。线程可以实现很多操作，并且运行速度非常快，运作相互独立。

+ GPU有上千上万的核数。GPU的每一个核都运行缓慢，能执行的操作没有CPU多。
+ GPU的核不能独立运行，它们需要共同协作。因为GPU有大量的核，当你需要同时执行操作时，它的并行能力很强。

CPU 跟 GPU 还有一点需要指明的是内存的概念。

+ CPU 有高速缓存但是相对比较小，而且CPU 的大部分内存都是依赖于系统内存。在一台典型的消费级台式机上- RAM -的容量可能有8，16或者32GB。
+ GPU 中内置了 RAM。 GPU 与系统RAM通信时，会带来产重的性能瓶颈，因此GPU 基本上拥有自己相对较大的内存。Titan XP 它的本地内存有12个GB。GPU 也有它自己的缓存系统，所以在 GPU的12个 G和GPU 核之间有多级缓存。它跟 CPU 的多层缓存类似。

![image-20220416232211582](https://raw.githubusercontent.com/verfallen/cs231n-2017-notes/main/img/image-20220416232211582.png)

## GPU 的并行能力

CPU 对于通信处理来说是友好的。而 GPU 更擅长于处理高度并行的算法其中最典型的的就是就是处理矩阵乘法。

下图中，左图是一个 A*B 的矩阵，乘以一个 B * C 的矩阵，最后得到一个规模为 A*C的输出矩阵。对于输出矩阵来说，每一个元素都是第一个矩阵的某一行与第二个矩阵的某一列的点积的结果。这些点积运算都是相对独立的。想象一下，将输出矩阵中每个元素并行计算，并且所有的计算都是两个向量点积的运算。实际上就是从两个输入矩阵的不同位置进行读取数据，然后进行点积，填到对应位置上。这是一个典型的适用于CPU解决的问题。

如果使用CPU来进行矩阵乘法的运算，可能会进行串行计算，就是对每个元素一个个进行计算。现在，CPU拥有多核，也可是进行矢量运算，但是针对并行任务，CPU通常表现得更好。特别是当矩阵规模非常庞大的时候。

<img src="https://raw.githubusercontent.com/verfallen/cs231n-2017-notes/main/img/image-20220418000318804.png" alt="image-20220418000318804"  />

## GPU 计算架构

可以在 GPU 上写出可以直接运行的程序。NVIDIA 有个叫做 CUDA 的抽象代码，可以让你写出类 C的代码，并且可以在 GPU上直接执行。但是想要写出高性能
并且充分发挥 GPU 性能的 CUDA 代码，是很困难的。你必须非常谨慎地管理内存结构，并且不遗漏任何一个高速缓存，以及分支误预测等等。所以自己编写高性能的 CUDA 代码是非常困难的。

NVIDIA 开源了很多的库，这些库实现了通用的计算语言，可以用来实现高度优化。举个例子，NVIDIA 有个叫做 **cuBLAS** 的库，可以**实现各种各样的矩阵乘法**，并且矩阵操作都是被高度优化的，可以在 GPU 上很好地运行，非常接近硬件利用率的理论峰值。还有一个叫做**cuDNN** 的库可以**实现卷积，前向和反向传播，批量归一化，循环神经网络等**。还有另一种语叫做**openCL**，这种语言在深度学习中更加普及**。它不仅可以在 NVIDIA GPU 上运行，还可以在 AMD 以及 CPU 上运行。**但是没有人花费大量的精力优化 openCL，所以 openCL 在性能上并没有 CUDA 好。

## CPU和GPU性能表现

将Intel 的 CPU8-14 和当时性能最好的GPU Pascal Titan X做一下性能比较，下图是测试的结果。更详细的结果可以 [参考这里。](https://github.com/jcjohnson/cnn-benchmarks)对于VGG 16/19 和不同层数的ResNet ，**对于同样的计算任务，CPU耗时是 GPU的65到75倍。**这个测试对CPU有些不公平，因为没有压榨出CPU的最大性能。只是在CPU直接安装运行了torch。



![image-20220418004740182](https://raw.githubusercontent.com/verfallen/cs231n-2017-notes/main/img/image-20220418004740182.png)

另一个有趣的结果是,比较了卷积优化的 cuDNN 库和没有经过优化的 直接以 CUDA 写成的代码，在相同的网络相同的硬件相同的 Deep Learning 框架上，可以在图表中看到大约有3 倍的速度差距。也就是说优化过的 cuDNN比原生 CUDA版代码快这么多。所以一般来说只要你在 GPU 上写代码,你就应该使用 cuDNN.

![image-20220418163640703](https://raw.githubusercontent.com/verfallen/cs231n-2017-notes/main/img/image-20220418163640703.png)

## CPU和GPU的通信

在训练网络的时候，你的模型可能存储在 GPU 上，但是庞大的数据集却存储在机械硬盘或者是固态硬盘上。从硬盘中读取数据的很可能成为训练速度的瓶颈。因为 GPU 非常快，它计算正向反向传播的速度非常快，从硬盘上串行地读取数据会拖慢训练速度，这会让训练变慢。

有一些解决的方法：如果数据集比较小，你可以把整个数据集读到内存中。或者数据集不小但是你有台内存足够大的服务器，也可以这么干。或者装固态硬盘替换掉机械硬盘，提升读取数据速度。

另一种常用的思路是：**在 CPU 上使用多线程来预读数据**。把数据读出内存或者读出硬盘存储到内存上，这样就能连续不断地将缓存数据高效地送给 GPU。这种方法不太容易实现，因为 GPU 太快了，如果不能及时把数据发送给 GPU，仅仅读取数据这一项就会拖慢整个训练过程。

## 关于CPU和GPU的提问

**<u>Q：在编写代码时，如何有效避免上述提到的问题？</u>**

A：从软件层面来说，最有效的事情就是 **设定CPU预读数据**。比如用多个 CPU 线程从硬盘中读取数据。GPU 在运算的同时，CPU 的后台线程从硬盘中取数据，主线程等待这些工作完成，在它们之间做一些同步化，让整个流程并行起来。如果你使用了下面要讲到的这些深度学习框架，它们已经替你完成了这部分操作。

# 深度学习框架

去年讲这门课的时候，主要讲的是caffe，torch，Theano 和 TensorFlow。Tensorflow那时候刚发布，还没有得到很广泛的使用。但在过去一年里，Tensorflow已经变得非常流行了，它可能是大多数人使用的主要框架，这是很大的变化。各种各样的新框架如雨后春笋般发布出来。尤其是 Caffe2 和 Pytorch 这两个来自Facebook 的新框架，百度有 Paddle ，微软有 CNTK，亚马逊用MXNet，还有很多其他的框架。

从图上可以发现很有趣的一点是，第一代深度学习框架大多是由学术界完成的。Caffe 来自伯克利，torch 起初由纽约大学维护，后来是与 Facebook 合作维护。Theano 主要由蒙特利尔大学实现。但是下一代深度学习框架全部由工业界产生。Facebook 做了 Caffe2 和 PyTorch，Google 做了 Tensorflow。过去几年里，一个有趣的转变发生了。这些想法从学术界转移到了产业，产业界提出了强大的框架来进行工作。

![image-20220418203043362](https://raw.githubusercontent.com/verfallen/cs231n-2017-notes/main/img/image-20220418203043362.png)

## 计算图

无论何时你进行深度学习，都要考虑构建计算图来计算目标函数。在线性分类器中，数据 X和权重 W进行矩阵乘法，可能使用 hinge loss 损失函数来计算损失，可能会再加一些正则项，将这些不同的操作拼起来成为一些图结构，叫做**计算图**。

![image-20220418203535149](https://raw.githubusercontent.com/verfallen/cs231n-2017-notes/main/img/image-20220418203535149.png)

在大型神经网络中，计算图结构非常复杂。有很多不同的层，不同的激活函数，不同的权重，在这个非常复杂的图中。而且你使用像神经图灵机这样的东西，
得到这些非常疯狂的计算图。因为它们实在是太庞杂了，你甚至不能将其画出来。所以深度学习框架意义重大。

<img src="https://raw.githubusercontent.com/verfallen/cs231n-2017-notes/main/img/image-20220418203637611.png" alt="image-20220418203637611"  />

## 深度学习框架的优点

有三点：

+ 可以轻易构建大型计算图
+ 容易计算梯度
+ 可以在GPU上高效运行

针对第一点，我们前面已经说过，在大型神经网络中，计算图结构非常复杂，构建这样的结构耗时耗力。我们希望框架能够自动做好这一步。

针对第二点，使用深度学习总需要计算损失在权重方向上的梯度，我们希望框架能够自动计算梯度，而不是自己去编写计算梯度的代码、

针对第三点，我们已经知道GPU有强大的并行计算能力，所以我们希望代码能在GPU上高效运行。不需要额外关注硬件细节。

## 计算图实现的不同方式

举个例子，这里有三个输入 X，Y，Z，我们要结合x和 Y来生成 A，然后再结合 a和 z来生成 b，最后我们要对 b 进行求和操作，将值传给最终的结果 C。

![image-20220418210829591](https://raw.githubusercontent.com/verfallen/cs231n-2017-notes/main/img/image-20220418210829591.png)


下图是使用NumPy实现这个计算图的代码，但是NumPy 只能在CPU端运行，无法利用GPU来进行加速。而且，使用NumPy，必须要自己计算梯度。这很麻烦，所以大部分深度学习框架的目标是 **编写像前向传播的代码，但是能够在GPU上运行，并且自动计算梯度**。

![image-20220418210937579](https://raw.githubusercontent.com/verfallen/cs231n-2017-notes/main/img/image-20220418210937579.png)

这是一个在 TensorFlow 中实现相同计算图的例子。其中前向传播的代码与NumPy 相似。但是 TensorFlow 中，只写了一行代码用来计算梯度（见下图），十分方便。

![image-20220418213040259](https://raw.githubusercontent.com/verfallen/cs231n-2017-notes/main/img/image-20220418213040259.png)

TensorFlow的运行可以CPU和GPU上进行切换。如下图所示，使用 `tf.device()`来实现。左图是在CPU上运行，右图是在GPU上运行。

<img src="https://raw.githubusercontent.com/verfallen/cs231n-2017-notes/main/img/image-20220418213245756.png" alt="image-20220418213245756" style="zoom: 67%;" /><img src="https://raw.githubusercontent.com/verfallen/cs231n-2017-notes/main/img/image-20220418213234659.png" alt="image-20220418213234659" style="zoom: 67%;" />

用 PyTorch 实现相同的计算图，代码看起来也差不多。首先定义变量开始构建一个计算图（红色框圈住的部分），然后进行前向传播（黄色框圈住的部分），之后计算梯度（蓝色框圈住的部分）。

![image-20220418214050739](https://raw.githubusercontent.com/verfallen/cs231n-2017-notes/main/img/image-20220418214050739.png)

在 PyTorch 中切换到 GPU 非常容易，只需要在运行计算之前把数据都转换成 CUDA 数据类型即可。

![image-20220418214504011](https://raw.githubusercontent.com/verfallen/cs231n-2017-notes/main/img/image-20220418214504011.png)

通过观察完整的代码可以知道，TensorFlow 和 PyTorch 的代码在前向传播中与Numpy 看起来几乎完全一样。这是因为Numpy 有一个很好的接口，它非常容易用来一起工作。

# TensorFlow 

以下的内容，我们使用一个两层的全连接ReLU 网络作为示例来讲解。输入随机数据，损失使用L2欧氏距离。这个示例并没有做什么有用的事情，仅仅为了讲解用。

从下图可以看出，代码可以分为两个阶段。

1. 定义计算图，这个计算图会运行多次
2. 将数据输入到计算图中

这个TensorFlow 非常通用的一种模式，**首先构建计算图，然后重复利用运行图模型**。

![image-20220419115258468](https://raw.githubusercontent.com/verfallen/cs231n-2017-notes/main/img/image-20220419115258468.png)

## 构建计算图

下图是详细分析构建计算图的部分（红色框的部分）。

**黄色框部分的是构建`placeholder`对象，**我们定义了 X和 Y，W1和w2并且创建了这些变量的`tf.placeholder` 对象。这些变量会成为图的输入结点，这些结点又是图中的入口结点。当我运行图模型时会输入数据，就是将它们放到我们计算图中的输入槽中，这跟内存分配没有一点相似之处，我们只是为计算图建立了输入槽，然后我们用这些输入槽在对应的符号变量上执行各种 TensorFlow 操作以便构建。

**蓝色框部分是进行前向传播，计算y的预测值和L2距离损失。**我们做了一个矩阵乘法，然后使用`tf.maximum` 来实现 ReLU 的非线性特性。接着另用矩阵乘法来计算y的预测值。然后使用用基本的张量运算来计算欧氏距离，计算目标值 y 和预测值之间的 L2损失。需要注意的是，这里的代码没有做任何实质上的运算，只是建立计算图结构来告诉 Tensor Flow当输入真实数据时怎样执行。

**绿色框部分是计算梯度，**在这个示例中就是让 Tensor Flow 去计算损失值在w1和w2方向上的梯度。这里同样没有进行实际的计算，它只是在计算图中加入额外的操作，让计算图算出梯度。

![image-20220419115451727](https://raw.githubusercontent.com/verfallen/cs231n-2017-notes/main/img/image-20220419115451727.png)

## 运行计算图

下面是将运行计算图部分拆分后的示意图。此时我们已经完成了计算图的构建，

**红色框部分是进入一个 `session` 运行图**

**黄色框部分是建立具体的数据**。一旦进入session 之后，就要建立具体的数据来输入到计算图。通常，TensorFlow 希望接收 Numpy 数组类型的数据。所以这个我们使用Numpy 数组建立具体的x,w1,w2,y，并将他们用字典类型进行存储。

**蓝色框部分是真正运行计算图**。我们使用了session.run() 来运行，在这个例子中，我们想知道`loss`,`grad_w1`,`grad_w2`这些值，我们将他们用字典参数的方式传入到计算图中，然后进行运行并计算出需要的值。在最后一行，将out进行解构之后，可以得到Numpy 数组类型的具体数组。

![image-20220419120656421](https://raw.githubusercontent.com/verfallen/cs231n-2017-notes/main/img/image-20220419120656421.png)

## 训练网络

至此，我们完成了一次正向和反向的传播运算。如果想要训练网络，需要添加如下红色框圈住的代码。我们用循环实现多次运行计算图，每次循环分为四步走。

1. 调用`session.run()` 计算损失和梯度
2. 结构输出，得到计算出的值
3. 更新当前权重值

![image-20220419160811876](https://raw.githubusercontent.com/verfallen/cs231n-2017-notes/main/img/202204191609476.png)

如果运行上述的代码并画出损失曲线，会得到这样一张图。随着迭代次数的增加，损失越来越小。网络训练得越来越好。

![image-20220419161219669](https://raw.githubusercontent.com/verfallen/cs231n-2017-notes/main/img/202204191612705.png)

## 优化

但这里有一个问题，当我们每次执行计算图进行前向传播时，我们将权重输入到计算图中，当计算图执行完毕我们会得到梯度值。权重是以Numpy 的形式保存，这意味着我们每次运行图的时候，我们必须复制权重，从Numpy数组格式中到 TehsorFlow中的格式才能得到梯度。输出的梯度又需要转为Numpy 格式才能进行更新权重的计算。这看起来不是个大问题但是，我们曾谈到 CPU 钳 GPU 之间的传输瓶颈，要在 CPU 和 GPU 之间进行数据复制非常耗费资源，所以如果你的网络非常大，权重值和梯度值非常多的时候，这样做就很耗费资源并且很慢。

**TensorFlow 的解决方法就是将w1和 w2 定义为TensorFlow 的变量保存。**变量可以一直存在于计算图中，也正因为如此，需要告诉TensorFlow 它是怎样初始化的。下图展示了将w1和w2 初始化为变量的代码。使用 `tf.random_normal()` 进行初始化操作，同样地，这里不是真正初始化，只是告诉框架到这里该如何做。

![image-20220419164901978](https://raw.githubusercontent.com/verfallen/cs231n-2017-notes/main/img/202204191649018.png)

然后更**新权重的部分也要进行改变。**之前我们在计算图之外进行定义权重，计算梯度，以Numpy array 的形式更新权重，在下一次的迭代中使用更新后的权重参数。现在权重被保存在计算图中，更新权重也要在计算图中进行操作。因此，我们使用 `assign` 函数，它能改变计算图中变量的值。这些值在计算图的迭代过程中一直存在。

![image-20220419194843882](https://raw.githubusercontent.com/verfallen/cs231n-2017-notes/main/img/202204191949610.png)

![image-20220419200202878](https://raw.githubusercontent.com/verfallen/cs231n-2017-notes/main/img/202204192002018.png)

然后开始运行计算图，首先进行了一个全局参数的初始化。然后进行迭代。

如果这时打印loss 曲线会发现，loss 是平的，并没有下降。这是为什么呢？

因为TensorFlow 太聪明了，它只执行了你要求结果所必须的操作。在上述代码中，我们只要求计算loss值。仔细观察你会发现，loss 并不依赖new_w1 和 new_w2，也就是说，loss 的计算直接使用了初始的 w1和w2，所以造成了loss 不下降的问题。我们必须明确地告诉 TensorFlow用new _w1和new_w2来执行操作。

直觉上，我们会把 new_w1 和 new_w2 作为输出，这样就能解决loss 不下降的问题。但是这里还有一个问题这些new_W1和new_W2都是非常大的tensor，当我们告诉 TensorFlow 我们需要这些输出时，我们每次迭代，就会在GPU和CPU之间不断重复复制数据的操作。

这里有一个小技巧，我们在图中添加一个仿制节点，让仿制节点依赖于new_w1和new_w2。当我们执行计算图时，我们同时计算loss 和这个仿制节点，仿制节点并不返回任何值，当我们执行了更新操作后，我们使用了更新的权重参数值。

![image-20220419201206221](https://raw.githubusercontent.com/verfallen/cs231n-2017-notes/main/img/202204192012280.png)

## 提问

**<u>Q：为什么不把x和y放入计算图，而把它们设置为Numpy结构？</u>**
A：在上述例子中，每个迭代都重复使用了同样 X和 Y，可以把它们放入计算图中。大多实际情况下，x和y是数据集的mini batch ,它们在每次迭代中是变化的，因此不希望把它们放在计算图中。

**<u>Q：关于updates</u>**
A：updates 并不是一个真的数值，它返回的是空。但是updates 依赖于new_w1，new_w2，它们存储在GPU中，所以我们可以在GPU中直接更新new_w1和new_w2这些值，不需要进行多余的复制操作。

<u>**Q：`tf.group`返回None吗？**</u>
A：它返回了TensorFlow 的内部节点操作，我们需要这些节点操作来构建图，当在`session.run` 执行图时,想要从`updates`中得到具体值，然后它返回`None`。

**<u>Q：为什么loss 是一个值，updates 返回None?</u>**
A：这是updates 工作的方式。loss 之所以是一个值，是因为我们告诉 TensorFlow 想要一个tensor时，就得到一个具体的值。updates 可以看做一种特殊的数据类型，它返回None。

## 优化器

先在我们想要执行不同赋值操作时，需要利用`tf.group `方法，这个方法有点奇怪，TensorFlow 有一个便捷方式来实现，叫做**优化器**。

这里使用了`tf.train.GradientDescentOptimizer()` 这个函数，传入学习率这个参数值，使用RMS prop优化方法，实际有很多其他不同的优化算法。然后调用` optimizer.minimize` 来最小化损失函数，通过这个调用可以知道W1和w2在默认情况下被标记为可训练，因此在这个`optimizer.minimize` 里,会进入计算图并在计算图中添加节点计算w1和w2的损失梯度然后它执行更新、分组、分配，但是最终返回更新值。如果仔细查看代码他们它内部实际上使用了`tf.group`，跟我们之前的代码很相似。

当在循环中运行计算图，采用相同的模式来计算损失值和更新值。

![image-20220420144606280](https://raw.githubusercontent.com/verfallen/cs231n-2017-notes/main/img/202204201447164.png)

**<u>Q：`tf.global_variables_initializer()` 是什么?</u>**
A：这个是用来初始化w1和w2，这些变量都存在于这个计算图中，所以我们需要用到这个。当我们创建 tf 的变量时，使用来了`tf.randomnormal`，`tf.global_variables_initializer`就是让
`tf.random_normal` 进行运行并生成具体的值来初始化这些变量。

## Loss

计算损失的方法有很多，可以使用我们自己的张量来精确的计算损失，TensorFlow 也给出很多便利的函数，能帮住计算一些常见的神经网络的相关结果。
在下面的例子中，我们使用了`tf.losses.mean_squared_error() `计算 L2 损失。

<img src="https://raw.githubusercontent.com/verfallen/cs231n-2017-notes/main/img/202204201505883.png" alt="image-20220420150536789" style="zoom: 50%;" />

## Layers

另一个繁琐的地方是我们必须明确定义输入，定义权重 ，然后像使用矩阵乘法一样，在正向传播中将它们链接在一起。在这个例子中我们其实没有在网络层中放入偏差，在实际的训练中我们还必须初始化偏差，我们必须让偏差保持正确的形式，才能进行计算。因此不得不传播矩阵乘法的输出偏差，这种写法很繁琐。一旦使用卷积层，批量规范层或者其他类型的层，更麻烦了。TensorFlow 的高级库可以处理这些细节。其中一个例子就是 **`tf.layers`**。

看下面的示例，在代码中这是显示地声明了 x和y，然后定义了h 是一个全连接网络，把输入数据x，单元数H作为参数，定义网络中的激活函数为 `tf.nn.relu`，也就是说这个层中的激活函数为 ReLu函数。这一行代码实际上做了很多事情，它设置了 w1和 b1，它为变量设置了正确的形状，这些变量存在于计算图中，对我们来说一可以说是隐藏的。它使用`xavie_initializer` 来为这些变量建立一个初始化策略。

可以看到这里调用了两次`tf.layers`去建立了模型，但是不需要自己来处理这些细节。`tf.contrib.layer` 并不是这里唯一的方式，还有很多其他的库。

![image-20220420151755391](https://raw.githubusercontent.com/verfallen/cs231n-2017-notes/main/img/202204201517488.png)

**<u>Q：`xavier_initializer` 的默认设置是不是一个特定的分布？</u>**
A：我确定它有一些默认设置，但不确定具体是什么，需要查看一下具体的相关文档。这是一个合理的分布策略，如果实际上运行这段代码，它的收敛速度比前一个要快得多，因为它的初始化取值更好。

## Keras

Keras是一个非常方便的 API，它建立在 Tensor Flow 的基础之上，并且在后端处理建立的计算图。Keras 也支持 Theano 作为后端。

下图是一个Keras的例子。

红色框部分构建了一个序列层的模型，黄色框部分构建了优化器对象，绿色框部分调用了`model.compile` ,就是在这里后端建立了计算图。蓝色框部分，调用`model.fit`自动完成了整个训练过程。这期间，不需要知道后端具体是怎么工作的。

![image-20220420160954984](https://raw.githubusercontent.com/verfallen/cs231n-2017-notes/main/img/202204201609067.png)

## 其他基于TensorFlow 的高级封装

+ Keras (https://keras.io/) 
+ TFLearn (http://tflearn.org/) 
+ TensorLayer (http://tensorlayer.readthedocs.io/en/latest/
+ tf.layers (https://www.tensorflow.org/api_docs/python/tf/layers) 
+ TF-Slim (https://github.com/tensorflow/models/tree/master/inception/inception/slim)
+ tf.contrib.learn (https://www.tensorflow.org/get_started/tflearn) 
+ Pretty Tensor (https://github.com/google/prettytensor) 
+ Sonnet (https://github.com/deepmind/sonnet)

上述都是一些基于TensorFlow 的高级封装，其中 keras 和 TFLearn 是第三方库，tf.layers，TF-Slim，tf.contrib.learn 都是TensorFlow 自带的，功能也各不相同。Pretty Tensor 来自于Google 内部，但是它不在TensorFlow 的框架内。Sonnet 来自于DeepMind 团队。这些框架之间不能很好地兼容，只是提供不同的选择。

## TensorBoard

TensorBoard 可以帮助添加一些指示性代码，画出训练过程中的loss 曲线等。

<img src="https://raw.githubusercontent.com/verfallen/cs231n-2017-notes/main/img/202204201627646.png" alt="image-20220420162713574" style="zoom: 50%;" />

# PyTorch

PyTorch 内部明确定义了三层抽象：

+ Tensor，张量，就像 Numpy数组一样，只是一种最基本的数组但它可以在 GPU 上运行。
+ Variable，变量，就是计算图中的节点，这些节点构成计算图，从而可以计算梯度等等
+ Module，模块，它是一个神经网络层可以将这些模组合起来
  建立一个大的网络

与TensorFlow 对比，可以将张量视为TensorFlow 中的Numpy 数组，Variable 对应 TensorFlow 中Tensor，Variable,PlaceHolder，它们在计算图中都是节点。Module 对应tf.layer，tf.slim，sonnet 或者其他高级封装的组合。

PyTorch 有一点需要注意，它的抽象层次很高，还有像Module 这样好用的抽象模块， 使用 `nn.module`基本已经足够了，不需要使用更高层的封装。

## Tensor 张量

就像之前说的那样，**PyTorch 中的张量就像是Numpy 的Array一样，只不过它可以在GPU上运行**。下图是只使用Tensor 完成的一个和前面例子相同的两层网络的代码。

红色框部分：为输入x,输出y,权重w1和w2 创建随机数据。

黄色框部分：进行前向传播，计算y的预测值和损失loss

蓝色框部分：进行反向传播，计算梯度

绿色框部分：根据计算出的梯度值，更新权重

![image-20220420221653684](https://raw.githubusercontent.com/verfallen/cs231n-2017-notes/main/img/image-20220420221653684.png)

如果要让Tensor 运行在GPU上，只需要使用不同的数据类型即可，将`dtype`修改为：

```
dtype = torch.cuda.FloatTensor
```

## Variable 变量

PyTorch 的第二层抽象就是 `Variable` 变量，就是计算图中的结点。一旦从张量转到变量，就建立了计算图，可以自动做梯度和一些其他计算。下图是将张量转变为变量的代码。（原来PPT的代码有bug，修改了一下）

红色框部分：创建随机数据，将他们设置为 Variable。拿x举例，x是一个变量，`x.data` 是一个`Tensor` 类型，`x.grad` 是另一个`Variable`，包含损失对x的梯度，`x.grad.data` 也是梯度，它是Tensor类型。在创建变量时，传入 `requires_grad`用来控制是否需要计算在该变量上的梯度。

黄色框部分：计算前向传播，与之前使用 `Tensor`的操作是一样的。PyTorch `Variable` 类型和 `Tensor` 类型有相同的API，因此任何在 `Tensor`上可以运行的代码，都可以使用`Variable` 进行替换。

蓝色框部分：反向传播，计算梯度。

绿色框部分：更新权重，梯度的值都保存在 `w1.grad.data`中。



![image-20220420230541759](https://raw.githubusercontent.com/verfallen/cs231n-2017-notes/main/img/image-20220420230541759.png)

### 自定义 Autograd 函数     

如果想要构建自己的计算梯度函数，可以通过为张量编写 前向和反向计算来实现。这里是一个实现ReLu 函数的例子。红色框是定义函数的部分，黄色框是在训练过程中使用自定义的梯度函数。

![image-20220420233130947](https://raw.githubusercontent.com/verfallen/cs231n-2017-notes/main/img/image-20220420233130947.png)

## Module

PyTorch 的第三层抽象就是 `Module`。它等价于TensorFlow 中的高级封装。不同的是，PyTorch 只用这一个封装器。下图是一个使用Module 的实例。

红色框部分：定义模型为一些层的序列，一个线性层，ReLu 激活，再加一个线性层。

黄色框部分：定义损失函数，使用均方差损失。

蓝色框部分：从这里开始进入了迭代循环体中，我们在模型中进行前向传递，得到预测值，然后计算损失。

绿色框部分：调用 `loss.backward()` 自动计算梯度。

黑色框部分：在模型的所有参数上循环，显式地更新参数值。

![image-20220421002153175](https://raw.githubusercontent.com/verfallen/cs231n-2017-notes/main/img/image-20220421002153175.png)

### 优化器

PyTorch 提供了优化操作。下面是一个使用优化器的例子，只需要两步就可以使用优化器。

红色框部分：建立了一个`optimizer` 对象，这里使用的是Adam 优化，然后设置将学习率之类的超参数。

蓝色框部分：在计算了梯度之后，调用`optimizer.step` 即可更新模型中所有的参数

![image-20220421002828451](https://raw.githubusercontent.com/verfallen/cs231n-2017-notes/main/img/image-20220421002828451.png)

### 自定义Module

一个模块只是神经网络中的一层，它可以 包含其他的模块。下图是一个自定义的模块，重现了上述两层网络的例子。

红色框部分：自定义一个两层的网络，继承自 `torch.nn.Module`。

黄色框部分：是这个类的初始化方法，可以看到，该类中定义了两层的线性网络，对应类中的 `linear1` 和 `linear2`。

蓝色框部分：前向传播，在对 Variable 类型使用内部模块和autograd 操作，计算网络的输出。前向传播的部分具体包括：将输入作为一个变量，将其传给 `self.linear1`作为第一层，然后使用 autograd 操作 `clamp` 函数去计算 relu ,再将输出传给 linear2，得到预测值。

绿色框部分：建立并训练一个模型实例。



<img src="https://raw.githubusercontent.com/verfallen/cs231n-2017-notes/main/img/202204211347474.png" alt="image-20220421134747288" style="zoom: 33%;" />

## DataLoader 

DataLoader 可以帮助建立分批处理，重排数据，还可以使用多线程。所以DataLoader 可以打包数据，并提供一些接口。当你需要执行你自己的数据的时候，会需要编写自己的Dataset 类。这样可以从指定的来源，读取特殊类型的数据，然后用Dataloader 打包并训练。下图是一个使用DataLoder 的实例。

红色框部分： 创建DataLoader对象

黄色框部分：迭代DataLoader 对象，每次迭代的过程中产生一批数据，然后在其内部重排，多线程加载数据。

![image-20220421141841052](https://raw.githubusercontent.com/verfallen/cs231n-2017-notes/main/img/202204211418129.png)