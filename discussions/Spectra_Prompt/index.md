##### 图预训练模型

链路预测

对比学习

- node dropping
- edge perturbation
- attribute masking
- subgraph

模型扰动

## Inductive Graph Alignment Prompt: Bridging the Gap between Graph Pre-training and Inductive Fine-tuning From Spectral Perspective

https://doi.org/10.1145/3589334.3645620

解决的问题：迁移学习虽然已经提出了许多基于图提示的方法来弥补像语言提示那样的预训练和微调任务类型所造成的差距，但这些方法仍然忽略了图数据的多样性。现有的基于图提示的方法是在预训练和微调图兼容的假设下运行的，这意味着所有这些方法都是可转换的，只有当gnn在同一图上进行预训练和微调时，才能保证性能。在归纳设置下，预训练的gnn在微调图上可能具有次优性能，甚至具有负转移。

探索图提示学习任务中的主要矛盾：

(1)图信号的差异

(2)图结构的差异

这些间隙表现为图谱理论中的节点特征扰动和谱空间错位

文章从谱图理论(数学推导)中提出了一种提示方法。

<img src=".\image-20240504221014683.png" alt="image-20240504221014683" style="zoom:50%;" />

`note`：**The framework of IGAP. We first align the graph signals and then we align the spectral space between the pre-train graph and fine-tune graph thus the pre-trained GNN model can be applied. A task-specific prompt is used to align the pre-train task and the fine-tune task. (basis)** 

思路：

1.首先从图谱理论中分析了图的预训练过程的本质：

已有的图预训练模型：

链路预测，子图对比，局部全局对比(是否子图)

统一框架：都是一个对比的训练过程，将信息分为正负样本进行对比：

<img src=".\image-20240501195137057.png" alt="image-20240501195137057" style="zoom:50%;" />

ps:最原版InfoNCE

<img src=".\image-20240505163202806.png" alt="image-20240505163202806" style="zoom:50%;" />

2.从多种任务中提取了一种loss函数，从这个loss函数推导出了图预训练学习到的内容主要在信号低频区域。(理解是比较平滑)

推导过程：

先统一定义了一个loss函数的范式

将训练样本采集和训练过程数学公式化

以loss梯度下降的目标推导出最小化主要影响了低频段，最小的几个$\lambda$对应的特征向量



3.图信号提示：和all in one类似的数学推导过程

<img src=".\image-20240504221842240.png" alt="image-20240504221842240" style="zoom:50%;" />



4.图结构提示：图的结构间隙本质上是训练前图和微调图之间的频谱空间的错位

​	根据第2点对正交基向量进行微调：$𝑈_{𝑝𝑡_𝐾} = 𝑃_𝑡𝑈_{𝑓𝑡_K}$

<img src=".\image-20240410162714334.png" alt="image-20240410162714334" style="zoom:50%;" />

z是频域信号，v是特征向量，x是图信号，$\lambda$是特征值，F是扰动以后的矩阵，q是扰动信号

𝜆𝑠𝑝,𝑖 is the 𝑖-smallest the eigenvalue of 𝐹𝑠𝑝 , 𝜆𝑑𝑡,𝑖 is the 𝑖-smallest the singular value of 𝐹𝑑𝑡and 𝜆𝑠𝑝,𝑖 ≪ 𝜆𝑑𝑡,𝑗 

<img src=".\image-20240410164144553.png" alt="image-20240410164144553" style="zoom:33%;" />

ptK是pre-train的前K个，ftK是fine-tune的前K个

<img src=".\image-20240503195047460.png" alt="image-20240503195047460" style="zoom:50%;" />

解决的问题：从频域的角度来寻找图提示学习的方法。

<img src=".\image-20240501195117295.png" alt="image-20240501195117295" style="zoom:50%;" />

频域上计算比在时域，在图上主要是空域，计算来得方便得多。



主要思路：把GCN的频域卷积的公式：带到一个基于对比学习的loss函数来推导出学习到的东西主要在低频段，然后对最小K个特征向量进行Prompt。
