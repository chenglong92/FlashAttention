# FlashAttention
Flash Attention Code Study for Large Language Model(LLM).

# 大模型算子
## 1. Attention
&emsp;从此前GPT-3 on Pytorch的测试结果可以看出Attention主要包含matmul->Dropout->Softmax->Mask->Matmul,其中时间占比并不是matmul最高,Dropout, Mask和Softmax占比也相当明显.
![](@attachment/Clipboard_2023-10-14-10-53-52.png)
#### (1) Standard Attention
&emsp;输入矩阵$\mathbf{Q}$, $\mathbf{K}$, $\mathbf{V}$,大小均为$N \times d$,且初始存储在HBM上,其中$N$代表sequence length,$d$代表head dimension.
&emsp;Step 1: 从HBM上Load $\mathbf{Q}$, $\mathbf{K}$(按照block方式),在运算单元上完成$\mathbf{S}=\mathbf{Q}\mathbf{K}^T$,进而将矩阵$\mathbf{S}$写入HBM中存储;
&emsp;Step 2: 从HBM读取矩阵$\mathbf{S}$,进而在运算单元计算$\mathbf{P}=softmax(\mathbf{S})$,将矩阵$\mathbf{P}$写入HBM;
&emsp;Step 3: 从HBM中Load矩阵$\mathbf{P}$和$\mathbf{V}$,然后完成计算$\mathbf{O}=\mathbf{PV}$,进而将矩阵$\mathbf{O}$写入HBM中,返回Attention矩阵$\mathbf{O}$.
&emsp;从上面流程可以看到,矩阵$\mathbf{S}$和$\mathbf{P}$矩阵均为中间生成矩阵,相比于原生输入矩阵在HBM中本身占有的存储,这两个矩阵涉及跨越内存层级搬运,且总矩阵大小为$N*N$.
&emsp;其中softmax的计算公式如下(这里向量$\mathbf{x}$大小是$B \times 1$):
$$m(\mathbf{x}) = \max_{i} \ x_i, \qquad f(\mathbf{x}) = e^{\mathbf{x}-m(\mathbf{x})}, \qquad l(\mathbf{x})=\sum_i {f(x_i)}, \qquad softmax(\mathbf{x}) = f(\mathbf{x})/l(\mathbf{x})$$
#### (2) Flash Attention 1.0
&emsp;综合而言,FA的核心思想包含两个: (a) 在前向和后向采用Tiling切分Softmax/score矩阵；(b) 在后向中采用重复计算("以算换存").
&emsp;此时的softmax公式在Tiling意义下则等价变换成(以二分块为例):
* Step 1: $\mathbf{x} = [\mathbf{x^{(1)}} \ \mathbf{x^{(2)}}]$,将原始向量$\mathbf{x}$分裂为两个Block;
* Step 2: $m(\mathbf{x}) = \max_{i} (m(\mathbf(x^{(1)})), m(\mathbf(x^{(2)})))$等价;
* Step 3: $f(\mathbf{x}) = [ e^{m(\mathbf{x^{(1)}})-m(\mathbf{x)}}f(\mathbf{x^{(1)}}) \qquad e^{m(\mathbf{x^{(2)}})-m(\mathbf{x)}}f(\mathbf{x^{(2)}}) ]$;
* Step 4: $l(\mathbf{x}) = l([\mathbf{x^{(1)}} \  \mathbf{x^{(2)}}]) = e^{m(\mathbf{x^{(1)}})-m(\mathbf{x)}}l(\mathbf{x^{(1)}})+e^{m(\mathbf{x^{(2)}})-m(\mathbf{x)}}l(\mathbf{x^{(2)}})$;
* Step 5: 计算$softmax(\mathbf{x}) = f(\mathbf{x}) / l(\mathbf{x})$.
&emsp;将以上过程展开,看计算步骤:
假设: Tr = Tc = 4,则循环以上Tiling的计算步骤实际上是先向下行循环,此时第一列的O仅仅涉及本块单元信息.
![](@attachment/Clipboard_2023-10-15-11-25-28.png)

_______
第一列:  $i = 1, ..., Tr$, $j=1$
$\mathbf{S}_{i1}=\mathbf{Q}_i{\mathbf{K}_1}^T$ ,&emsp; &emsp; (size: $Br\times Bc$),
$\mathbf{m}_{i1} = rowmax(\mathbf{S}_{i1})$ ,&emsp; &emsp; (size: $Br$),
$\mathbf{P}_{i1} = exp(\mathbf{S}_{i1}-\mathbf{m}_{i1})$,&emsp; &emsp; (size: $Br\times Bc$),
$\mathbf{L}_{i1} = rowsum(\mathbf{P}_{i1})$,&emsp; &emsp; (size: $Br$)
$\mathbf{O}_i = (\mathbf{P}_{i1} \mathbf{V}_1)/\mathbf{L}_{i1}$,&emsp; &emsp; (size: $Br \times d$)
$\mathbf{L}_i = \mathbf{L}_{i1}$, $\mathbf{m}_i = \mathbf{m}_{i1}$,&emsp; &emsp; (size: $Br$)
* 例如: 当i=1时
$\mathbf{S}_{11}=\mathbf{Q}_1{\mathbf{K}_1}^T$ ,&emsp; &emsp; (size: $Br\times Bc$),
$\mathbf{m}_{11} = rowmax(\mathbf{S}_{11})$ ,&emsp; &emsp; (size: $Br$),
$\mathbf{P}_{11} = exp(\mathbf{S}_{11}-\mathbf{m}_{11})$,&emsp; &emsp; (size: $Br\times Bc$),
$\mathbf{L}_{11} = rowsum(\mathbf{P}_{11})$,&emsp; &emsp; (size: $Br$)
$\mathbf{O}_1 = (\mathbf{P}_{11} \mathbf{V}_1)/\mathbf{L}_{11}$,&emsp; &emsp; (size: $Br \times d$)
$\mathbf{L}_1 = \mathbf{L}_{11}$, $\mathbf{m}_1 = \mathbf{m}_{11}$,&emsp; &emsp; (size: $Br$)

第二列: $i = 1, ..., Tr$, $j=1$
$\mathbf{S}_{i2}=\mathbf{Q}_i{\mathbf{K}_2}^T$ ,&emsp; &emsp; (size: $Br\times Bc$),
$\mathbf{m}_{i2} = rowmax(\mathbf{S}_{i2})$ ,&emsp; &emsp; (size: $Br$),
$\mathbf{P}_{i2} = exp(\mathbf{S}_{i2}-\mathbf{m}_{i2})$,&emsp; &emsp; (size: $Br\times Bc$),
$\mathbf{L}_{i2} = rowsum(\mathbf{P}_{i2})$,&emsp; &emsp; (size: $Br$)
$\mathbf{O}_i = (exp(\mathbf{m}_{i1}) \mathbf{P}_{i1} \mathbf{V}_1 + exp(\mathbf{m}_{i2}) \mathbf{P}_{i2} \mathbf{V}_2)/(exp(\mathbf{m}_{i1})\mathbf{L}_{i1}+exp(\mathbf{m}_{i2})\mathbf{L}_{i2})$,&emsp; &emsp; (size: $Br \times d$)
* 例如: 当i=1时
$\mathbf{S}_{12}=\mathbf{Q}_1{\mathbf{K}_2}^T$ ,&emsp; &emsp; (size: $Br\times Bc$),
$\mathbf{m}_{12} = rowmax(\mathbf{S}_{12})$ ,&emsp; &emsp; (size: $Br$),
$\mathbf{P}_{12} = exp(\mathbf{S}_{12}-\mathbf{m}_{12})$,&emsp; &emsp; (size: $Br\times Bc$),
$\mathbf{L}_{12} = rowsum(\mathbf{P}_{12})$,&emsp; &emsp; (size: $Br$)
$\mathbf{O}_1 = (exp(\mathbf{m}_{11}) \mathbf{P}_{11} \mathbf{V}_1 + exp(\mathbf{m}_{12}) \mathbf{P}_{12} \mathbf{V}_2)/(exp(\mathbf{m}_{11})\mathbf{L}_{11}+exp(\mathbf{m}_{12})\mathbf{L}_{12})$,&emsp; &emsp; (size: $Br \times d$)


或者对于任意块$(i,j0)$,$i \in [1,Tr]$, $j0 \in [1,Tc]$其Attention Matrix$((Br*Tr)*d)$的输出结果:
$$O_i = \frac{\sum_{j=1}^{j=j0} {(e^{\mathbf{S}_{ij}}\mathbf{V}_j)}} {rowsum({\sum_{j=1}^{j=j0} {e^{\mathbf{S}_{ij}}}})}$$
