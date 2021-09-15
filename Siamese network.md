# Siamese	network

参考文章：[Triplet Network, Triplet Loss及其tensorflow实现](https://www.jianshu.com/p/b1188c9f5fd2)

​					[Triplet Loss and Online Triplet Mining in TensorFlow](https://omoindrot.github.io/triplet-loss#batch-hard-strategy)

​					FaceNet: A Unified Embedding for Face Recognition

​					Deep metric learning using Triplet network

# 1 背景

由一个外国连体婴儿产生的，在英语中为“连体，孪生”的意思。

其中，在小样本数据中，例如门禁系统，在多数情况下不用神经网络，因为训练样本小，所以很容易造成识别率差的结果。那么我们如何将极小的输入样本进行分类呢？

# 2 神经网络的特点

有两种神经网络：一个是孪生神经网络，一个是伪孪生神经网络。

<img src="https://pic3.zhimg.com/80/v2-5070e28622a2f3ee9e3cb5d2259fae86_720w.jpg" alt="img" style="zoom:50%;" />

上图为孪生神经网络，就是他们共享权值。

==在用途上，==孪生神经网络适合于**“比较类似的”**，伪孪生适合有**一定差别的**。

# 3 模型

小明（x1）![](C:\Users\49252\AppData\Roaming\Typora\typora-user-images\image-20201230181701896.png)

相当于将输入x1去encoding了，不进行softmax，只进行到flatten展开这一步，比如说产生了180个数。那么我们将这个==模型==直接输入x2（这里相当于共享权值），它又会得到180个数。==这里使用到了one-shot思想。==

## 3.1 模型的训练

* loss function的选用---**triplet loss**

  <img src="https://upload-images.jianshu.io/upload_images/7915866-8d4fef33b3672750.jpg?imageMogr2/auto-orient/strip|imageView2/2/w/720/format/webp" alt="img" style="zoom: 67%;" />

训练的目标就是**让相同标签的人的空间距离更近，让不同标签的空间距离更远。**

也就是$||f(xa)-f(xp)||^2≤||f(xa)-f(xn)||^2$，当然我们不可能让两个距离相等，除非输入完全相同的两个图片，所以我们为了减小图像的“贪婪值”（这里借助强化学习的概念，自己定义的），给他增加一个阈值，即：
$$
L = ||f(xa)-f(xp)||^2-||f(xa)-f(xn)||^2+\alpha≤0
$$
这时候他的loss定义为==max（L，0）==

## 3.2 训练数据的选择

我们该如何选择三元组数据，即anchor、positive、negative。

这里当我们使用triplet loss时，当我们总是选择两个相似程度很小的图片时，我们获得的结果便是显而易见的，则神经网络本身就无法学到有用的知识。我们把将这种称为easy-triplet，在更多情况下，我们一般定义使公式（2）成立的图。

==而hard-triplet数据，定义为不加阈值成立，而加阈值就不成立的图。==

在facenet中，hard-triplet效果最好。

# 4 衍生

论文是《**Deep** **metric learning using Triplet network**》，输入是三个，一个正例+两个负例，或者一个负例+两个正例，训练的目标是让相同类别间的距离尽可能的小，让不同类别间的距离尽可能的大。

<img src="https://pic2.zhimg.com/80/v2-8502a1627d1752e5b398ac93d8f93d4d_720w.jpg" alt="img" style="zoom: 67%;" />

# 5 关于我的思考

我们先看对象，分别是**欠烧态，正烧态，过烧态**

![image-20201230191622686](C:\Users\49252\AppData\Roaming\Typora\typora-user-images\image-20201230191622686.png)

我们收集的图像相对清晰，已经无法分别出物料区，火焰区，还有煤粉区。

这三种图片是非常相似的，只是在中间一部分的形状，还有部分的色彩分布有差别。

这时候我认为可以使用衍生的triplet network模型。

其次，考虑到模型不够精准的可能性，我们使用**k-means**进行图像分割。

<img src="C:\Users\49252\AppData\Roaming\Typora\typora-user-images\image-20201230191533846.png" alt="image-20201230191533846" style="zoom:150%;" />

**将图像进行分割后分别训练，或者是感受野逐渐增大去训练，最后将结果去融合。****

loss函数的选取，也就是代替上面的范数距离，我想选取的是关于颜色的距离，其中常见的颜色距离有**LAB颜色空间、改进的加权欧式距离等**（==待看==）

最后，我们将最后一层选取不同的分类器就行实验比较。



margin改变