* 使用100维glove词向量进行embedding
* 或使用100维由SDS数据集训练的word2vec
* 未使用textrank4zh包
* 动手建立textrank计算图
* 结果低于使用textrank4zh包


### Results
| Dataset | 词向量 | Acc | Pr | Re | F1 |
| - | - | - | - | - | - |
| SDS | word2vec(100d) | 0.573857 | 0.403497 | 0.258084 | 0.297611 |
| SDS | glove(100d) | 0.574851 | 0.414335 | 0.255691 | 0.298083 |
| ADS | word2vec(100d) | 0.596491 | 0.342550 | 0.209114 | 0.243548 |
| ADS | glove(100d) | 0.586007 | 0.313854 | 0.220836| 0.243746|
