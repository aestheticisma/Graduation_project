* 使用100维glove词向量进行embedding
* 或使用100维由SDS数据集训练的word2vec
* 未使用textrank4zh包
* 动手建立textrank计算图
* 结果低于使用textrank4zh包


### Results
| 词向量 | Acc | Pr | Re | F1 |
| - | - | - | - | - |
| word2vec(100d) | 0.539273 | 0.370716 | 0.300348 | 0.315420 |
| glove(100d) | 0.558557 | 0.398852 | 0.327947 | 0.342571 |
