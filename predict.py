import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd

MODEL_DIR = "./bert_news_classifier/checkpoint-3939"  # 最新版
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_DIR,
    from_tf=False,
    #ignore_mismatched_sizes=True
)

# 加载标签（必须和训练时完全一致）
id2label = model.config.id2label
label2id = model.config.label2id

print("模型标签映射:", id2label)

# 预测函数
def predict(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=64
    )
    model.eval()
    with torch.no_grad():
        out = model(**inputs)
    idx = out.logits.argmax().item()
    return id2label[idx]

# 交互
print("="*50)
print("已加载新闻标题分类模型。。。")
print("输入 quit 退出")
print("="*50)

while True:
    msg = input("\n请输入新闻标题：")
    if msg.lower() in ["quit","exit","q"]:
        break
    if not msg:
        continue
    print("预测分类：", predict(msg))