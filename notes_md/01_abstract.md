# 计算机视觉

**计算机视觉**顾名思义就是针对视觉数据的研究。

世界上的摄像头比人更多，每天都产生有超级超级多的视觉数据。2015年的研究数据表示，估计在2017年，互联网大约80%的流量都是视频。从一个纯粹的数据占比的角度来说，在互联网上传播的大多都是视觉数据。那么如何使用算法来开发和利用这些数据呢？视觉数据存在问题，它们真的很难理解，有时我们把视觉数据称为互联网的暗物质，将它与物理学中的暗物质类比。在物理学中，暗物质占整个宇宙质量的一大部分，我们能知道是因为在各种天体上存在万有引力，我们不能直接观察到它。互联网上的视觉数据也是一样的，它们占网络传输数据的大部分，但是算法很难去探知并理解这些视觉数据到底是些什么。

另一个统计实例来自 YouTube，大概每秒钟世界上发生的事情其中有长达五小时的内容会被上传到 YouTube。如果想要给视频分类，为观众推荐相关视频，通过投放广告来赚钱 ，我们希望使用技术，让机器可以沉浸式观测并且理解这些视觉数据。
所以这个计算机视觉领域是一个跨学科的领域。

# 计算机视觉发展历程

## 生物视觉的历史

视觉的历史可以追溯到很久很久以前。大约5亿4千三百万年前，地球几乎完全被水覆盖，那时候只有少量的物种在海洋中游荡，没什么生机。动物不怎么活跃，没有眼睛。有猎物路过时，它们就抓来充饥。但是在大约5亿4千万年前发生了一件意义重大的事。通过对化石的研究，动物学家发现在短短的一千万年里，动物的物种数量爆炸式增长，从少数几种发展到成百上千，被称为物种大爆炸。是什么造成了这个奇怪的现象呢？有很多相关理论，但这件事仍然是未解之谜。后来，一位名叫 安德鲁 帕克的澳大利亚动物学家提出了一种很有说服力的理论。他发现，距今5亿4千万年前，第一次有动物进化出了眼睛，是**视力功能的出现促使了物种数量的爆炸。**动物们一旦有了视力，生物开始变得更积极主动，捕食者追赶猎物，而猎物要逃避捕食者。视力的出现开启了一场进化的竞赛，物种们为了生存必须尽快地演化。这就是动物拥有视觉的开端。今天，视觉成为了动物，尤其是有智慧的动物最重要的感知系统 。人类的大脑皮层中几乎一半的神经元与视觉有关。这项最重要的感知系统，使我们可以生存，工作，运动，操作器物，沟通，娱乐，等等。以上讲的是生物的视觉。

<img src="https://raw.githubusercontent.com/verfallen/gallery/master/cs231n-2017-notes/image-20220503232513053.png" alt="image-20220503232513053" style="zoom: 67%;" />

## 机器视觉的历史

那么人类让机器获得视觉或者说照相机的历史是什么样的？我们现在已知最早的相机，要追溯到17 世纪文艺复兴时期的暗箱。是一种通过小孔成像的相机，与动物早期的眼睛非常相似。通过小孔接收光线，后面的平板收集信息并投影成像。如今照相机已经非常普及了。摄像头可以说是手机或者其它装置上最常用的传感器之一。

<img src="https://raw.githubusercontent.com/verfallen/gallery/master/cs231n-2017-notes/image-20220503235108223.png" alt="image-20220503235108223" style="zoom: 67%;" />

同时，生物学家也开始研究视觉的机理。其中最具影响力的要数五六十年代休伯尔和维泽尔使用电生理学的研究。这个研究启发了计算机视觉的研究。
他们提出的问题是哺乳动物的视觉处理机制是怎样的。他们选择与人类相似的猫类来进行研究，他们将电极插进主要控制猫视觉的后脑上的初级视觉皮层，然后观察何种刺激会引起视觉皮层神经的激烈反应。
他们发现猫大脑的初级视觉皮层有各种各样的细胞，简单的细胞会在有向边沿着特定方向移动时有反应。复杂细胞对光线方向和移动有反应。总的来说，**视觉处理是始于视觉世界的简单结构**。视觉信息变化，大脑建立了复杂的路径，直到它可以识别更为复杂的视觉世界。

<img src="https://raw.githubusercontent.com/verfallen/gallery/master/cs231n-2017-notes/image-20220503233423079.png" alt="image-20220503233423079" style="zoom:67%;" />

## 计算机视觉历史

计算机视觉的历史也从60 年代初开始的，Block World 是由 Larry Roberts 出版的一部作品，被公认为是计算机视觉的第一篇博士论文。其中视觉世界被简化为简单的几何形状，目的是容易识别，并重建这些形状。

![image-20220504000308604](https://raw.githubusercontent.com/verfallen/gallery/master/cs231n-2017-notes/image-20220504000308604.png)

1966年，有一个MIT暑期项目叫做The Summer Vision Project。它如今已经非常著名。这个项目的目标是有高效利用暑期时间来构建视觉系统的重要组成部分。五十年过去了，计算机视觉领域已经那个夏季项目发展成为全球数千名研究人员的领域。我们还没有弄清楚人类视力的原理，但是我们已经能够处理一些简单的问题。这个领域已经成为人工智能领域最重要和发展最快的领域之一。

另一件不得不提是是一个叫做David Marr的人，他是麻省理工学院视觉科学家。在70年代后期，他撰写了一本非常有影响力的书——《VISION》。内容是关于他对视觉的理解，以及如何处理计算机视觉开发 和让计算机识别视觉世界的算法。他在书中指出，为了拍摄一幅图像并最终全面的3D视觉表现，必须经历几个过程。**第一个过程叫做 原始草图**，这个阶段由边缘，端点，虚拟线条，曲线，边界等组成。这是受到了神经学家的的启发，Hubel 和 Wiesel 告诉我们，视觉处理的早期阶段有很多关于像边缘的简单结构。**下一过程是2.5 维草图**，将表面，深度信息，层等拼凑在一起，然后最终将所有内容放在一起成为一个3d 模型。

<img src="https://raw.githubusercontent.com/verfallen/gallery/master/cs231n-2017-notes/image-20220504001111140.png" alt="image-20220504001111140" style="zoom:67%;" />

这是一个进入视觉领域非常直观的方式——考虑如何解构视觉信息。
七十年代，有一个开创性的工作组提出来一个问题，我们如何越过简单的块状世界识别或表示现实世界的对象？斯坦福大学的两个科学家组提出了类似的想法，一个被称为广义圆柱体 ，一个被称为图形结构。基本思想是每个对象都由简单的几何图单位组成。
例如，一个人可以通过广义的圆柱形形状拼接在一起。或者也可以由一些关键元素按照不同的间距组合在一起。所以表示方法是将物体的复杂结构简约成一个几何体，有更简单的形状和几何结构。

<img src="https://raw.githubusercontent.com/verfallen/gallery/master/cs231n-2017-notes/image-20220504002636434.png" alt="image-20220504002636434" style="zoom:67%;" />

80年代 ，David Lowe 思考如何重建或者识别由简单的物体结构组成的视觉空间。他尝试识别通过线和边缘构建的剃须刀。其中大部分都是直线以及直线之间的组合。
从60年代到80年代，我们都试图去解决目标识别问题。在这个过程中，人们开始思考，如果目标识别太难了，那可以先做**目标分割，就是把一张图片中的像素点归类到有意义的区域，**我们可能不知道这些像素点组合到一起是一个人型，但是我们可以把属于人的像素点从背景中抠出来，这个过程叫做图像分割。

<img src="https://raw.githubusercontent.com/verfallen/gallery/master/cs231n-2017-notes/image-20220504003033708.png" alt="image-20220504003033708" style="zoom:67%;" />

还有另一个问题先于其他的计算机视觉问题有进展，那就是**面部检测**。大概在 1999-2000年，机器学习技术 ，特别是统计机器学习方法开始加速发展。出现了一些方法，比如支持向量机模型，boosting 方法，图模型，包括最初的神经网络。Paul Viola 和 Michael Jones 使用AdaBoost 算法进行实时面部检测。这项工作是在计算机芯片还是非常慢的2001年完成的，但是他们还是能够实现准实时的面部检测。在论文发表的第5年，也就是2006年，富士推出了第一个实现实时面部检测的数码相机。这是从基础科学研究到实际应用的一个快速的转化。

<img src="https://raw.githubusercontent.com/verfallen/gallery/master/cs231n-2017-notes/image-20220504003433742.png" alt="image-20220504003433742" style="zoom:67%;" />

怎样才能够做到更好的目标识别呢？在90年代末到2000年，目标识别的思想是基于特征来做。这是由David Lowe 发明的，叫做SIFI特征。例如这里有一个stop标识，去匹配另一个stop标识是非常困难的。因为有很多变化的因素，如相机的角度，遮挡视角，光线，及目标自身的内在变化。但是可以得到一些启发，标的某些特征可以在变化中具有表现性和不变性。**可以观察这些不变的特征来进行识别**。所以目标识别的首要任务是在目标上确认这些关键的特征，然在在目标物体上匹配这些特征，这比匹配整个目标要容易得多。
下图是一张他论文中的图，图中显示了一个stop 标志几个SIFT特征，这些特征与另一个stop 标识的 SIFT 特征相匹配。

![image-20220504003923986](https://raw.githubusercontent.com/verfallen/gallery/master/cs231n-2017-notes/image-20220504003923986.png)

使用相同的图片特征，怎么识别整幅图的场景呢？这里有一个例子叫做空间金字塔匹配。背后的思想是图片有各种特征，这些特征可以告诉我们图片是什么类型的，是风景，厨房，还是公路等等。这个算法从图片各部分和不同像素中抽取特征，并把他们放在一起作为特征描述符然后在特征描述符上做支持向量机。

![image-20220504004759016](https://raw.githubusercontent.com/verfallen/gallery/master/cs231n-2017-notes/image-20220504004759016.png)

还有一项类似的研究。把这些特征放在一起以后，研究如何在实际图片中比较合理地合成人体姿态，辨认人体姿态。这方面一个工作被称为方向梯度直方图，
另一个被称为 可变性不见模型。

![image-20220504004903196](https://raw.githubusercontent.com/verfallen/gallery/master/cs231n-2017-notes/image-20220504004903196.png)

## 图片数据集的历史

可以看到随着互联网的发展，计算机视觉的研究拥有更好的数据。计算机视觉在21 世纪早期指出了一个非常重要的基本问题，就是目标识别。
我们一直在提目标识别，但是直到21世纪早期，我们才开始真正拥有标注的数据集能够衡量在目标识别方面取得的成果。

其中最有影响力的标注数据集之一叫做**PASCAL Visual Object Challenge**，这个数据集由20个类别组成。数每个种类有成千上万张图片，不同领域的团队开发算法来与测试数据集做对抗训练看有没有优化。下图右边列举了从2007到2012年在基准数据集上检测图像中的20 种目标的检测效果，可以看到一直在稳步提升。

![image-20220504005151842](https://raw.githubusercontent.com/verfallen/gallery/master/cs231n-2017-notes/image-20220504005151842.png)



普林斯顿和斯坦福中的一批人提出了一个更为困难的问题，我们是否具备了识别真实世界中每一个物体的能力？这个问题也是由机器学习的一个现象驱动的，大部分的机器学习算法都很有可能在训练的过程中过拟合。部分原因是可视化数据很复杂，正是因为它们太复杂了，模型往往维数比较高，也就有一堆参数要调优。当训练数据量不够时，很快就会产生过拟合现象，泛化能力就很差。

因此就是有两方面的动力，一是我们单纯就是想识别自然世界中的万物，二是要回归机器学习克。服机器学习的瓶颈--过拟合问题、我们开展了一个叫做ImageNet 的项目，这个项目中汇集所有能找到的图片，包含世界万物，组建一个尽可能大的数据集 。因此耗时3年才完成这个项目。最开始，从网上下载上亿张图片，用一个叫做WordNet 的字典来排序。这个字典里有上万个物体类别。接下来去排序，清洗，给图片打上标签。ImageNet有将近1500万甚至4000万多的图片，分成22000的物体或场景。这很有可能是当时 AI领域最大的数据集，它将目标检测算法的发展推到一个新的高度。

<img src="https://raw.githubusercontent.com/verfallen/gallery/master/cs231n-2017-notes/image-20220504005835209.png" alt="image-20220504005835209"  />

从2009年开始，ImageNet团队组织了一场国际比赛，叫做 ImageNet 大规模视觉识别竞赛。这个比赛中有一个筛选更为严格的测试集，总共140万的目标(图像)有1000种目标类别。这个数据集为了测试算法的图像分类，识别的正确率。如果一个算法能输出概率最大的5个类别其中有正确的对象，就认为是识别成功。下面是ImageNet挑战赛2010-2015年图像分类结果的统计，X轴表示比赛的年份，Y轴表示错误率。可以看到错误率正在稳步下降。到2012年，错误率已经低到和人类识别一样。虽然我们没有完全解决目标识别问题，但这已经是很大的进步了。在图上有一个很特别的时刻，是2012年。
在挑战赛的前两年，错误率在25% 左右。而在2012年，误差率下降了近10% ，达到了16%。那一年的获奖算法是一种卷积神经网络模型，也就是现在被熟知的深度学习。在2012年，CNN模型展示了强大的模型容量和能力，在计算机视觉和其他领域都取得了巨大的进步。

![image-20220504010002709](https://raw.githubusercontent.com/verfallen/gallery/master/cs231n-2017-notes/image-20220504010002709.png)

# 视觉识别任务

## 图像分类

**图像分类就是让算法接收一张图作为输入，从固定的类别集合中选出该图像所属的类别**。基本的分类器可以应用在很多地方，比如应用于食物识别，显示食物中的卡路里。

## 目标检测

目标检测与一整幅图像的分类有小小的区别。**背景中有一辆车，在目标检测中我们要画出边界框标示出在图像中的位置。**

<img src="https://raw.githubusercontent.com/verfallen/gallery/master/cs231n-2017-notes/image-20220506225908526.png" alt="image-20220506225908526" style="zoom:67%;" />

## 图像摘要

**图像摘要就是给定一幅图像，生成一段句子来描述这幅图像。**这看起来与图像分类完全不同，但是为了图像识别而开发的工具也可以在这里用到。

# 大规模视觉识别比赛

前面已经在图像数据集部分已经提过 ImageNet，在这个比赛上获得优胜的都是算法中的佼佼者。

2011年 提出的算法中，包含了很多分层。首先计算一些特征，然后计算局部不变特征，经过一些池化操作再经过多层处理，最终将结果描述符传递给线性 SVM。
这里仍然使用了层次结构，仍然检测边界，有不变特征的概念。

![image-20220506230659744](https://raw.githubusercontent.com/verfallen/gallery/master/cs231n-2017-notes/image-20220506230659744.png)

2012年提出的算法实现了真正的突破。来自多伦多大学的 Jeff Hinton 小组和他当时的博士生Ale Krizhevsky和Ilya Sutskever共同创造了这个七层卷积神经网络，现在以AlexNet 而闻名，那时也被叫做Supervision。从那年开始所有ImageNet的获胜者都被神经网络包揽了。而且这些网络的发展趋势是越来越深。AlexNet 这个神经网络有七或八层，取决于需要计算的精确度。

![image-20220506231255213](https://raw.githubusercontent.com/verfallen/gallery/master/cs231n-2017-notes/image-20220506231255213.png)

2014年我们有了一些更深的网络，它们是来自Google 的GoogleNet 和 来自牛津大学的VGG网络。

![image-20220506231546743](https://raw.githubusercontent.com/verfallen/gallery/master/cs231n-2017-notes/image-20220506231546743.png)

2015 年优胜的算法网络更深，它被称作残差网络有152层。从此以后只要能将层数加到200 以上，性能总能得到一点提高。不过也有可能造成 GPU 内存溢出。

![image-20220506231627619](https://raw.githubusercontent.com/verfallen/gallery/master/cs231n-2017-notes/image-20220506231627619.png)

## 卷积神经网络的提出

总的来说，卷积神经网络是在2012 年取得了突破，但是这个算法却不是在2012年才被发明的。与卷积神经网络有关的其中一项基础性工作是由 Yann LeCun和他的合作伙伴于90年代在 Bell 实验室完成的。在1998 年他们使用卷积神经网络进行数字识别，他们希望将它用于自动识别手写的支票以及邮局识别地址。他们建立这个卷积神经网络，可以识别数字或字母。**该网络的结构看起来和2012年的 AlexNet非常相似。**

可以看到，网络中有很多卷积层和下采样，以及全连接层。如果只看这两幅图像，这个2012 年的网络与90年代的网络结构相似。

![image-20220506232038619](https://raw.githubusercontent.com/verfallen/gallery/master/cs231n-2017-notes/image-20220506232038619.png)

## 神经网络迟迟不流行的原因

神经网络的算法早在90年代就已经提出，为什么现在才变得流行呢？原因有二：

1. 计算力，90年代计算机的算力无法支撑神经网络。根据摩尔定律，从90 年代至今计算机芯片晶体管的数量已经增长了好几个数量级。我们还有了具有超高的并行计算能力的 GPU，非常适合进行卷神经网络模型这类高强度的计算。
2. 数据集。你需要提供给模型大量带标签的图像，以便于它们最终能实现更好的效果。然而在互联网普及之前，要收集种类繁多的数据集非常困难。2010 之后
   ，像 PASCAL和 ImageNet这样高质量的数据集，都获得了数量级的增长。有了这些庞大的数据集，才能实现更强大的模型。

# 视觉识别领域的其他挑战

## 动作识别

如果给定一个含有一些人物的视频，怎样才能最好地识别其中人物的活动。

随着增强现实和虚拟现实的发展，新技术和新型传感器的出现，我们将提出更多新的，有趣的，富有挑战性的问题。这是视觉实验室的一个例子，该数据集叫做视觉基因组。在这里不仅要框定出无题，还要描述整幅图像。这其中要包含目标的身份，目标之间的关系，目标的属性，场景动作等等。这种表达允许使用简单分类去捕捉真实世界中的东西。

![image-20220508095019763](https://raw.githubusercontent.com/verfallen/gallery/master/cs231n-2017-notes/image-20220508095019763.png)

## 图像描述

在这个实验中找了一些人，向他们展示图片半秒钟。即使图片一闪而过，人们依然能够写出很长的描述段落来讲述关于这张图片的故事。
仔细想想，这很了不起。仅仅看了这图片半秒，一个人就能够说出，这是两组人的游戏或竞技，左边的人在扔东西，发生在户外，因为似乎看到了草地等等细节。
如果花费更多时间来看这张图片，甚至有人能写出一篇小说。关于图像中的人是谁，以及为什么在那玩这种游戏。借助于储备知识和过往经验，还可以继续构想出其它情节。

这在某种程度上是计算机视觉领域的圣杯，以一种非常丰富而深刻的方式去理解一张图片的故事。尽管在过去几年里已经取得了巨大的进步，但是要夺取依然还有很长的路要走。

![image-20220508100100754](https://raw.githubusercontent.com/verfallen/gallery/master/cs231n-2017-notes/image-20220508100100754.png)