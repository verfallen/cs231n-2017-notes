# 图像分类

前面已经提过，**图像分类就是让算法接收一张图作为输入，并且有一些固定分类标签的集合，从固定的类别集合中选出该图像所属的类别。**图像分类是一个计算机视觉中的核心任务。这对于人类来说很简单，因为人类大脑里的视觉系统天生就是来做这些视觉识别任务，但是**对于一台机器来说真的是一个非常非常困难的问题**。

## 计算机图像分类难点

### 语义鸿沟

当一个计算机看着这些图片的时候，它看到的是什么？它没有一只猫咪这样的的整体概念，计算机呈现图片的方式其实就是数字，举例来说，下面这幅图就是800*600的像素，每一个像素由三个数字表示分别对应红、绿、蓝三个值。所以对于计算机来说，图像就是一个巨大的数字阵列，**很难从中提取猫咪的特性**，我们把这个问题称为**语义鸿沟**。

![image-20220510105538172](https://raw.githubusercontent.com/verfallen/gallery/master/cs231n-2017-notes/image-20220510105538172.png)

### 视觉偏差

如果稍微改变一下图片，整个像素网格都会发生变化。比如还是同一样一只猫静坐着，如果我们把相机移动到另一边，那么这个巨大的像素网格中的每一个像素都会完全不同，但是它仍然代表着同样一只猫，同时我们的算法需要对这些变化鲁棒。

### 照明

场景中会有不同的照明条件，无论猫咪出现在怎样样的漆黑、皆暗的场景中，我们的算法也应该是鲁棒的。

![image-20220510105912679](https://raw.githubusercontent.com/verfallen/gallery/master/cs231n-2017-notes/image-20220510105912679.png)



### 变形

目标对象有时还会变形，比如猫咪可以有干奇百怪的姿势和位置，对于不同的形变情形我们的算法也应该是鲁棒的。

![image-20220510112200631](https://raw.githubusercontent.com/verfallen/gallery/master/cs231n-2017-notes/image-20220510112200631.png)

### 遮挡

也可能有遮挡的问题，比如可能仅看到猫咪的一部分,比如说脸甚至只看到尾巴尖儿从沙发垫下露出。我们的算法需要鲁棒地应对的一种情况

![image-20220510112428446](https://raw.githubusercontent.com/verfallen/gallery/master/cs231n-2017-notes/image-20220510112428446.png)

### 背景混乱

处理的时候还会有图片背景混乱的问题，比如可能一张图片的前面所显示的是一只猫，但是可能猫身上的纹理会和背景十分相似。

<img src="https://raw.githubusercontent.com/verfallen/gallery/master/cs231n-2017-notes/image-20220510112633063.png" alt="image-20220510112633063" style="zoom:67%;" />

### 类内差异

这张图看上去是一些猫但实际上每只猫都有视觉上的差异。猫有不同的外形、大小，颜色以及年龄。算法需要鲁棒地处理这些不同场景。

![image-20220510112716960](https://raw.githubusercontent.com/verfallen/gallery/master/cs231n-2017-notes/image-20220510112716960.png)

# 硬编码图像识别方法

对于猫来说，我们知道猫有耳朵、眼睛、嘴巴、鼻子。根据 Hubel 和 Wiesel 的研究我们了解到边缘对于视觉识别是十分重要的。所以我们可以尝试计算出图像的边缘，然后把边、角各种形状分类好，比如有三条线是这样相交的那这就可能是一个角点，比如猫耳朵有这样那样的角，诸如此类我们可以写一些规则来识别这些猫。

但是实际上呢这种算法并不是很好，有以下缺点：

+ 正确率不高，很容易出错
+ 不可拓展，如果已经有了一个猫的分类器，想要对另一种对象进行分类，需要重新制定规则，重新训练。

![image-20220510113356355](https://raw.githubusercontent.com/verfallen/gallery/master/cs231n-2017-notes/image-20220510113356355.png)

# 数据驱动的方法

我们想发明一些识别算法，可以拓展到识别世界上各种对象。基于此我们想到用数据驱动的方法，不写具体的分类规则来识别一只猫或者鱼，取而代之的是**我们从网上抓取数据，比如大量猫的图片数据集，大量飞机的图片数据集等等。有了数据集之后，我们训练机器来分类这些图片，机器会收集所有数据，用某种方式总结**
**然后生成一个模型，总结出识别出对象不同类别的核心方法，然后用这些模型来识别新的图片。**

模型会变成两个函数，一个是训练函数，接收图片和标签，然后输出模型。另一种是预测函数，输入图片，对图片标签进行预测。正是用了这种方法，在过去十几二十年间，图像识别领域的进步非常大。

![image-20220510115921590](https://raw.githubusercontent.com/verfallen/gallery/master/cs231n-2017-notes/image-20220510115921590.png)

# 最近邻

最简单的分类器也许是**最近邻方法，在训练机器过程中什么也不做，只是单纯记录所有的训练数据。在图片预测过程中，接收新的图片，在训练数据中寻找与新图片最相似的，然后基于此来给出一个标签。**

![image-20220510120410943](https://raw.githubusercontent.com/verfallen/gallery/master/cs231n-2017-notes/image-20220510120410943.png)

## 示例

以CIFAR 10 数据集为例，这个数据集给出10个不同的类别，有50000张训练图片，10000张测试图片。在这个数据集上使用最近邻分类器，下图给出与测试图片最相似的图片。可以看到，分类的效果不太好，但它仍然是一个好的示例。

![image-20220512094611737](https://github.com/verfallen/gallery/blob/master/cs231n-2017-notes/image-20220512094611737.png?raw=true)

##  测量图片的距离

### L1 距离

 L1距离，也称为**曼哈顿距离。假设训练数据和测试数据都是 4x4 的像素，它们的L1距离等于训练图片像素的值减去测试图片对应像素的值，然后取绝对值，最后将差值的绝对值全部相加**。下例子，两幅图像的L1距离是456。

![image-20220510121335927](https://raw.githubusercontent.com/verfallen/gallery/master/cs231n-2017-notes/image-20220510121335927.png)

### L2 距离

L2距离，也称为欧氏距离。取平方和的平方根作为距离。

### L1 距离 VS L2 距离

不同的距离度量在预测的空间里对底层的几何或拓扑结构做出不同的假设。

+ **距离图像**，L1距离形成的图像是一个方形。方形上的每一个点的 L1 上是与原点等距的，L2距离实际上是一个根据 L1 距离的这个围绕着原点的方形形成的圆。

  ![image-20220512100623236](https://raw.githubusercontent.com/verfallen/gallery/master/cs231n-2017-notes/image-20220512100623236.png)

+ **坐标系的影响**，L1 距离取决于你选择的坐标系统，如果转动坐标轴将会改变点之间的L1 距离，而改变坐标轴对 L2 距离毫无影响。

+ **决策边界**，采用不同的方式度量距离，形成的决策边界形状变化很大。采用L1距离度量形成的决策边界，倾向于跟随坐标轴，这是因为L1 取决于坐标轴的选择。L2 距离只是将决策边界自然放置在它应该存在的地方。

  ![image-20220512102359016](https://raw.githubusercontent.com/verfallen/gallery/master/cs231n-2017-notes/image-20220512102359016.png)

### 如何选择

如果特征向量中的一些值对任务有一些重要的意义，那么也许 L1可能更适合。但如果它只是某个空间中的一个通用向量，那么 L2 可能更合适。

## 代码实现

下图是最近邻方法对应的Python代码实现。可以看到，在训练函数中，只需要存储数据。在测试函数中，使用L1 距离，度量输入图像和训练图片的距离，在训练集中找到最相似的实例。

![image-20220510141416536](https://raw.githubusercontent.com/verfallen/gallery/master/cs231n-2017-notes/image-20220510141416536.png)

## 时间复杂度

在训练集中我们有n个实例，训练和测试的时间复杂度分别是多少呢？

**训练的时间是O(1)**，这是一个常量。因为在训练过程中，只做存储数据。**预测的时间复杂度是O(n)**，因为在预测中，需要将数据集的N个实例与测试图像进行比较，比较的时间与测试数据集的规模n成正比。

所以，最近邻的实际使用过程中，训练过程比较快，测试过程比较慢。**这与我们期望的不一致，我们希望测试过程速度快。**因为训练过程是在数据中心完成的，它可以负担起非常大的运算量，从而训练出一个优秀的分类器。在测试过程部署分类器时，希望能运行在手机，浏览器 或者其他低功耗设备上。这时希望分类器能够快速地运行，由此看来最近邻算法有点落后了。之后要介绍的 **卷积神经网络**正好相反，它们会花很多时间在训练上而在测试过程则非常快。

## 决策边界

下图是这个最近邻分类器的决策区域。训练集是二维平面中的这些点，点的颜色代表不同的类别，这里有五个类。对每个点来说，将计算这些训练数据中最近的实例，然后在这些点的背景上着色，标示出它的类标签。可以发现最近邻分类器是根据相邻的点来切割空间并进行着色。观察下图可以发现一个问题，图像的中部集中了大多数的绿点
但中心却有一个黄点，因为我们只计算最近的点，所以在绿色区域的中间这个黄点被分割成了独立一块。这其实并不太好，还有一个相似的情况。由于绿色区域像一根手指一样插入了蓝色区域，而这个点可能是噪声或者失真信号。

![image-20220511234340421](https://raw.githubusercontent.com/verfallen/gallery/master/cs231n-2017-notes/image-20220511234340421.png)

# K 近邻

K近邻算法，根据距离度量，找到最近的 K个点。然后用这些相邻中进行投票，根据投票预测出结果。也可以考虑使用距离加权再进行投票，最简单的方法还是进行多数投票。

## K近邻 VS 最近邻

这里我们用同样的数据集，分别画出K=1，K=3，K=5 的决策区域。可以看到K=3时可以看到，绿色点族中的黄色噪点不再会导致周围的区域被划分为黄色。由于使用多数投票，中间的整个区域都将被分类为绿色。原本伸入红色和蓝色区域也开始被平滑掉，在 K=5时蓝色和红色区域间的决策边界更加平滑好看。因此**使用K近邻分类器时，倾向于给 K赋一个比较大的值，这样会使决策边界更加平滑，从而得到更好的结果。**

（图中的白色区域代表，在这个区域内没有获得K近邻的投票，可以将其归为其中某一个类别。）

![image-20220512000206481](https://raw.githubusercontent.com/verfallen/gallery/master/cs231n-2017-notes/image-20220512000206481.png)

## 示例

在CIFAR 10 数据集上使用NN算法，效果如下。（绿色框表示分类正确，红色表示分类错误）可以看出， 表现效果不是很好，但如果我们使用一个更大的 K 值，包括前三名或前五名
甚至包括到所有的数据范围进行投票，算法会有更大的鲁棒性。

![image-20220512094805133](https://raw.githubusercontent.com/verfallen/gallery/master/cs231n-2017-notes/image-20220512094805133.png)

[这里](http://vision.stanford.edu/teaching/cs231n-demos/knn/) 是一个在线展示KNN分类的网页。

## 超参数设置

**超参数就是那些不一定能从训练数据中学到，需要提前为算法指定的参数。**实践证明，超参数的设置是依赖于具体问题的，**最简单的做法是为你的任务和数据尝试不同超参数的值，并找出最优的值。**

### 设置验证集

更常见的做法 就是**将数据分为三组，大部分数据作为训练集，然后建立一个验证集，一个测试集，在训练集上用不同超参来训练算法，在验证集上评估出表现最好的一组超参数，最后将其用于测试集。**必须要确保测试集数据受到严格的控制，最后一刻才接触测试集。

![image-20220512104909013](https://raw.githubusercontent.com/verfallen/gallery/master/cs231n-2017-notes/image-20220512104909013.png)

### 交叉验证

设定超参数的另一个策略就是交叉验证，**常用于小数据集。**在深度学习中不那么常用。
它的理念是将把整个数据集保留部分数据作为最后使用的测试集，对于剩余的数据，将其分成很多份，轮流将每一份都当做验证集。

下图是使用5 折交叉验证的示例图，先在前4 份上训练，在第5 份上验证。然后再次训练算法，在1、2、3、5份上训练，在第4 份上验证。但在深度学习中，训练本身非常消耗计算能力，因此不常用交叉验证。

![image-20220512105331573](https://raw.githubusercontent.com/verfallen/gallery/master/cs231n-2017-notes/image-20220512105331573.png)

经过交叉验证，可以得到下图。x 轴表示K近邻分类器中参数 K 的值， y轴表示不同 K在相同数据上的准确率。这个例子里我们用了5 折交叉验证，可以观察到算法在不同验证集上表现的方差，还能看到算法效果的分布。在这个例子中，k=7 时效果最好。

![image-20220512111822002](https://raw.githubusercontent.com/verfallen/gallery/master/cs231n-2017-notes/image-20220512111822002.png)

## 缺点

KNN在实际中，很少使用。

1. 测试时运算时间很长，与我们的需求不符。

2. L1距离和L2 距离不太适合用在图像比较上。举个例子，下面的几幅图像与原图的L2 距离是相同的（通过刻意地设置来达到L2距离相同），但是视觉感受大不相同。

   ![image-20220512112721064](https://raw.githubusercontent.com/verfallen/gallery/master/cs231n-2017-notes/image-20220512112721064.png)

3. 维度灾难，我们需要大量数据来覆盖整个空间。

### 维度灾难

K -近邻分类器，有点像是用训练数据把样本空间分成几块。这意味着如果希望分类器有好的效果，需要训练数据能密集地分布在空间中。我们需要很多数据，数据的量是维度的指数倍。指数倍的增长从来不是好消息，我们也根本不可能能拿到那么多的图片去密布这个高维的像素空间。

来看下图，点代表训练数据，颜色代表分类标签。如果在一维空间中，只需要4个点就能将空间覆盖，二位空间需要 4 的平方，也就是16个点才能覆盖整个空间。如果是在高维空间中，需要训练样本数会呈指数倍增长。因为 KNN对潜在的各种分布情况没有任何预设，唯一能让它正常工作的方法是在样本空间上有着密集的训练样本。

![image-20220512112913383](https://raw.githubusercontent.com/verfallen/gallery/master/cs231n-2017-notes/image-20220512112913383.png)
## 提问

**<u>Q：训练集和验证集的区别是什么?</u>**
A： 在K邻分类算法用于图像分类时，训练集就是一堆贴上标签的图片。训练时，记下标签
要给图像分类时，将图片与训练集的每一个元素比较，然后将与训练点最接近点的标签作为结果。算法会记住训练集中的所有样本。验证集的作用则是用其中的每个元素，与训练集中的元素比较，将它为依据，判定分类器在验证集上表现如何。这就是训练集和验证集的区别，算法可以看到训练集中的各个标签，但是不能直接看到验证集中每个元素的标签。

**<u>Q：测试集是否有可能不能很好地代表真实世界中的数据？</u>**
A：是的。统计学假设就是数据都互相独立，服从同一分布，在现实中不可能总是这样。因此，当创建数据集时需要特别注意。常用的做法是，使用完全相同的方法来收集数据，后期再将它们打乱，随机分成训练集和测试集。只要对整个数据集的划分是随机的，就可以避免这种问题。

# 线性分类器

## 神经网络的一个基础模块

**线性分类器是一个相当简单的算法，它是深度学习网络基本的构建块之一。**

深度学习的构建有时就像是堆积木一样。举个例子，我们想要做图像描述。输入一副图像，输出描述图像的句子。我们已经有了关注于图像的卷积神经网络，关注语言的循环神经网络，那么可以将这两个网络放在一起，就像是堆积木一样，将其一块进行训练，最终得到一个厉害的模型。所以深层神经网络就像是乐高玩具，线性分类器就像是这种巨型网络的基础模块。

## 参数模型

与KNN不同，线性分类器是一个参数方法。回想一下KNN的训练过程中是没有设置参数的，直接保存训练集，用于测试使用。**参数化的方法则是总结对训练数据的认知，把学到的知识保存到参数中。**在测试时，只需要这些参数即可，无需训练数据集，这让我们的模型更具有效率。甚至可以运行在像手机这样的小型设备上。

参数模型由两部分组成，以线性分类器为例，左边是一副图片，右边是给定的10个数字，对应CIFAR-10 中的10个类别的分数。中间的函数f中有两个参数，通常将输入数据称为$x$，设置的参数或权重，通常写作$w$,有时也写作$\theta$。如果函数的输出猫的分数更大，则表明输入x是猫的可能性更大。

![image-20220522232630994](https://raw.githubusercontent.com/verfallen/gallery/master/cs231n-2017-notes/image-20220522232630994.png)

## 线性分类器的结构

在深度学习中，整个模型最终就是要找出函数 $F$ 的正确结构，不同的函数形式就对应不同的神经网络结构，你可以用不同的方式来组合权重和数据，最简单的方式就是相乘，这就是一个线性分类器的形式，即：
$$
f(x,W)=Wx
$$
将上述式子中的维度写明，$x$是一个 $32×32×3$ 的向量，展开之后就是一个 $3072×1$ 的向量，输出对应10个类别，因此是一个 $10×1$ 的向量，则 $W$是一个 $10×3072$ 的向量。

通常情况下会加上一个偏置项 $b$，$b$是一个 $10×1$ 的常数向量。它不与训练数据交互，仅仅是给一些数据独立的偏好只。如果数据集不平衡，比如猫的数量多于狗的时候，那么与猫对应的偏差值就会比其他要高。
$$
f(x,W)=Wx+b
$$

## 如何理解线性分类器

### 线性分类器是一种模板匹配方法

举个例子，左边是输入的图像，假使它是 $2×2$的矩阵，有4个像素。总共有3个类别，则权重矩阵是 3×4 ，偏置是 3×1 的向量，计算之后输出的分数跟别对应猫，狗，船这三个类别。**==线性分类几乎可以理解为是一种模板匹配方法，权值矩阵中的每一行对应于图像的某个模板==**，根据输入矩阵行和列之间的乘积。计算这个点积让我们发现在这个类的模板和图像的像素之间的相似之处，偏差则给了每个类的偏移量。**

![image-20220523000410487](https://raw.githubusercontent.com/verfallen/gallery/master/cs231n-2017-notes/image-20220523000410487.png)

如果根据这个模版匹配的观点思考线性分类器，可以将权重矩阵的行向量还原回图像，也就是将这些模版可视化成图像，这展示了线性分类器理解的数据方式。

下图就是在CIFAR-10 数据集中训练得到的权重矩阵中的行向量对应于10 个类别相关的可视化结果。举例来说在图中最左边，飞机类别的模版似乎是由中间类似蓝色斑点状图形和蓝色背景组成，可以认为飞机的线性分类器可能在寻找蓝色图形和斑点状图形，然后这些特征让分类更倾向于飞机。或者看这个汽车的例子，在中间有一个红色斑点状物体以及在顶部是一个或许可能是挡风玻璃的蓝色斑点状物体，这看起有些奇怪，真的不像一辆车。

![image-20220523001532787](https://raw.githubusercontent.com/verfallen/gallery/master/cs231n-2017-notes/image-20220523001532787.png)

== **但是存在这样一个问题，线性分类器只能为一个类别学习一个模板**==，如果这个类别中的数据出现了变体，那么它将求这些不同图片的平均值放在分类器中，因为一个模版只能用来识别一种类别，就可能会发生奇怪的情况。比如，在马匹的分类器中，马儿看起来好像有两个头，一左一后。线性分类器只能尽其所能，因为它只允许学习每个类别的一个模版。==**当使用神经网络和更复杂的模型，可以得到更好的准确率，因为这些模型没有只能每个类别学习一个单独的模版的限制。**==

### 线性分类器是一个分类面

如果回到高维空间的概念，每张图片都是高维空间的一个点，那么 **==线性分类器就是在尝试画一个分类面，将一种分类与其他的类别区别开来==**。比如在训练中，线性分类器会尝试绘制这条蓝色直线，将飞机和其他类别区分开来。

![image-20220524221713203](https://raw.githubusercontent.com/verfallen/gallery/master/cs231n-2017-notes/image-20220524221713203.png)

当从高维空间的角度来考虑线性分类器，会再次出现问题。**==有些情况下无法绘制一条直线将不同类的样本完全分开。==**

下图就是回会让线性分类器失效的例子。有两类数据，红色和蓝色分别代表不同的类。最左边的例子，蓝色代表图像中像素的数目大于0且为奇数，红色代表像素数目大于0且为偶数。可以以看到，蓝色类别的样本在两个相反的象限上，无法绘制一条直线将其分开。中间的例子中，蓝色样本在4个象限中都存在。最右边的例子，蓝色样本存在于三个不同的想先，其他所有的点都是另一个类别。

![image-20220524222336367](https://raw.githubusercontent.com/verfallen/gallery/master/cs231n-2017-notes/image-20220524222336367.png)