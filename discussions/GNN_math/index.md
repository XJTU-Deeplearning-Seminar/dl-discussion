# GNN谱图理论简述

XJTU_zsy

`从代数角度分析GNN的工作原理`

- [GNN谱图理论简述](#gnn谱图理论简述)
- [谱图理论简述](#谱图理论简述)
  - [1.拉普拉斯算子简述](#1拉普拉斯算子简述)
    - [简介](#简介)
    - [傅里叶级数简述](#傅里叶级数简述)
    - [思想借鉴](#思想借鉴)
      - [1.定义图拉普拉斯算子](#1定义图拉普拉斯算子)
      - [2.图频域基底](#2图频域基底)
  - [傅里叶变换](#傅里叶变换)
  - [卷积和频域](#卷积和频域)
  - [对图信号处理](#对图信号处理)
    - [直观理解这个公式](#直观理解这个公式)
  - [reference](#reference)

# [谱图理论简述](#谱图理论简述)

`从代数角度分析GNN的工作原理`

## [1.拉普拉斯算子简述](#1.拉普拉斯算子简述)

### [简介](#简介)

我们知道拉普拉斯算子是一个二阶微分算子，$\Delta f = \nabla ^2 f = \Delta ·\Delta f $ 我们通常在计算函数的二阶导数时用到它。

定义上看：它反映的是函数变化率的变化率。

### [傅里叶级数简述](#傅里叶级数简述)

可以由$\Delta f = n^2f$这个特征方程推出一组正余弦函数$\mathscr{F}=\{1, cos(nx), sin(nx)\}$，这组函数$\mathscr{F}$构成函数$f(x) \in L_2$上的一组正交基，可以对函数$f(x) \in L_2$进行展开，也就是傅里叶级数展开。函数$f$的傅里叶级数展开可以理解为函数$f$在不同基函数下的投影，可以认为是$f$中的不同n(正余弦的频率系数)大小。

从而可以用$\phi(n)$来描述这个函数。

显然，这个$\phi(n)$的定义域是离散的。

进一步地，我们找到了一组基底$\mathscr{F}=\{1, cos(nx), sin(nx)\}$来描述一个函数空间$f(x) \in L_2$，使得我们对$f(x) \in L_2$的研究可以统一到一组基底上，为我们提供了一个全新的角度。

### [思想借鉴](#思想借鉴)

#### [1.定义图拉普拉斯算子](#1.定义图拉普拉斯算子)

图$\mathscr{G} = (X, A)$,A是邻接矩阵, 用u表示节点，节点的特征信息用$x=f(u)$表示，$X = f(U)$。

离散拉普拉斯算子

$∇_d^2f=f(x+1,y)+f(x−1,y)+f(x,y+1)+f(x,y−1)−4f(x,y)$

推广到图:

单个节点的拉普拉斯算子：
$$
\begin{align}
\Delta f(u_j)\\
&=\sum_{u_i \in \ neighbor(u_j)} f(u_j)-f(u_i)\\
&=degree(v)\times f(v)-\sum_{u_i \in \ neighbor(v)}f(u_i)\\
&=\sum_k A_{jk}\times f(u_j) - A_j *f(U)\\
(1)
\end{align}
$$
则所有节点：
$$
\begin{align}
\Delta f(U) \\
&= [\Delta f(u_1)\ \Delta f(u_2)\ ...]\\
&=[...(\sum_k A_{jk}\times f(u_j) - A_j *f(U))...]\\
&=[...(\sum_k A_{jk}\times f(u_j))...]-A*f(U)\\
&=D*f(U)-A*f(U)\\
&=(D-A)*f(U)\\
(2)
\end{align}
$$
其中D是度对角阵。*表示通常意义的内积。

#### [2.图频域基底](#2.图频域基底)

记$\bar{A} = D^{-\frac{1}{2}}(D-A)D^{-\frac{1}{2}}$其中$D^{-\frac{1}{2}}$为了对称归一化，显然$\bar{A}$是一个实对称矩阵，$\lambda \vec{x} = \bar{A} \vec{x}$可以求出它的|U|个特征向量和对应的非负特征值，这几个特征向量$\mathscr{VEC} = \{\vec{v_1},... \}$就构成了可以描述这个图的拓扑结构的基底，记$V^T$的列为$v_i$:
$$
\bar{A} = V\Lambda V^T\\
(3)
$$


## [傅里叶变换](#傅里叶变换)

在傅里叶级数的基础上，用复数欧拉公式可以较为容易推导出傅里叶变换为：
$$
F(ω)=F(f(t))=∫ _{−∞}^{+∞}f(t)e^{−jωt}dt\\
(4)
$$
就是把上面傅里叶级数的n从整数变到w实数范围以后，将原函数对各个基底作投影。

类似的：

定义图上傅里叶变换
$$
F(u_i) = F(f(u_i)) = Vx_i\\
(5)
$$
逆变换
$$
f(u_i) = F^{-1}(F(u_i)) = V^TVx_i\\
(6)
$$


## [卷积和频域](#卷积和频域)

卷积：单位脉冲函数和原函数卷积为原函数

信号处理：时不变系统中，处理作用于原函数，相当于处理作用于单位脉冲函数后和原函数卷积

卷积和频域：时域卷积等于频域相乘后逆变换。



## [对图信号处理](#对图信号处理)

冲击函数在图上就是每个节点施加一个constant([1,1,1....])。

在$\mathscr{H}()$下图信号X的输出Y，则：
$$
\begin{align}
Y \\
&= V^T((V\mathscr{H_0}(\Lambda)) (VX)),\\
\\
&把V\mathscr{H_0}()并为\mathscr{H}()\\
\\
&= V^T((\mathscr{H}(\Lambda)) (VX))\\
(7)
\end{align}
$$
对$\mathscr{H}(\Lambda)$作切比雪夫展开：
$$
\mathscr{H}(\Lambda) \approx \sum_{i=0}^K\theta_iT_k(\hat{\Lambda})，\\
\hat{\Lambda} = \frac{2\Lambda}{\lambda_{max}}-I_N是为了满足切比雪夫展开的条件\\
(8)
$$
用一阶近似并取得$\lambda_{max} = 2$(切比雪夫多项式$Tk$取$Tk(x) =\theta_0 + \theta_1x$)：
$$
\begin{align}
Y\\
&代入(7)(8)\\
&=\theta_0X+\theta_1V^T((\Lambda-I_N) (VX))\\
&= \theta_0 X-\theta_1 (D^{-\frac{1}{2}}AD^{-\frac{1}{2}}X)\\
&令\theta = \theta_0 = -\theta_1则:\\
&= \theta ((D^{-\frac{1}{2}}AD^{-\frac{1}{2}}+I_N)X)\\
&将D^{-\frac{1}{2}}AD^{-\frac{1}{2}}+I_N归一化到[0,1]:\\
&= \theta (((\hat{D}^{-\frac{1}{2}}\hat{A}\hat{D}^{-\frac{1}{2}}))X)\\
&总体来看，\\
&D^{-\frac{1}{2}}(D-A)D^{-\frac{1}{2}} --> \hat{D}^{-\frac{1}{2}}\hat{A}\hat{D}^{-\frac{1}{2}},\hat{A} = A + I_N, \hat{D_{ii}} = \sum_j \hat{A_{ij}}\\
(9)
\end{align}
$$
最后加上激活函数得到了单层GCN：
$$
Y= \sigma ((\hat{D}^{-\frac{1}{2}}\hat{A}\hat{D}^{-\frac{1}{2}})HW)\\
$$

### [直观理解这个公式](#直观理解这个公式)

聚合邻居节点的信息。

<img src=".\p2.png" alt="image-20240430182145629" style="zoom:50%;" />

<img src=".\p1.png" alt="image-20240430182214076" style="zoom:50%;" />

## [reference](#reference)

**Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering**

[ https://doi.org/10.48550/arXiv.1606.09375](https://doi.org/10.48550/arXiv.1606.09375)



[CS224W | Home (stanford.edu)](https://web.stanford.edu/class/cs224w/)



[GNN入门之路: 01.图拉普拉斯矩阵的定义、推导、性质、应用 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/368878987)