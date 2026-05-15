# 基于BERT的中文新闻标题分类
## 项目功能
中文新闻文本数据清洗与预处理
基于 BERT 预训练模型微调训练
单条新闻分类预测
## 数据来源
[今日头条新闻分类数据集](https://github.com/fateleak/toutiao-text-classfication-dataset)
## 快速使用
### 1.数据预处理
`python data_process.py`
### 2. 模型训练
`python train.py`
### 3. 新闻分类预测
`python predict.py`