## Results
| Bert | Acc | Pr | Re | F1 | epoch |
| -------------- | -------- | -------- | -------- | -------- | -------- |
| Bert(768d) | 0.698217 | 0.607618 | 0.445587 | 0.486075 | 20 |
| Bert(768d) | 0.691649 | 0.567332 | 0.512873 | 0.514327 | 100 |
| Bert(768d)+LSTM | 0.680601 | 0.549627 | 0.475951 | 0.493668 | 50 |
| Bert(768d)+LSTM+ATTENTION | 0.693587 | 0.572805 | 0.475973 | 0.501993 | 50 |
| Bert(768d)+TextCNN | 0.691978 | 0.575581 | 0.452505 | 0.486622 | 50 |

## Some Records
### 调用bert-as-service获取句向量
1. 安装bert-as-service
分为服务端和客户端
```bash
pip install bert-serving-server
pip install bert-serving-client
```
2. 下载Bert预训练模型
可以到[这里](https://github.com/google-research/bert#pre-trained-models)查看所有预训练模型。
我用的是`BERT-Base, Uncased`
3. 启动服务端程序
```bash
bert-serving-start -model_dir /tmp/english_L-12_H-768_A-12/ -num_worker=4
```
具体参数请看[介绍](https://github.com/hanxiao/bert-as-service#2-start-the-bert-service)

4. 模型定义均置于`main.py`(除Attention层定义外)

5. 运行时切换模型需注意修改输入张量的维度，详情见代码注释