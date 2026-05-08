import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HUGGINGFACE_HUB_CACHE"] = "./cache"
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from datasets import Dataset


df = pd.read_csv("toutiao_clean_data.csv")

df = df.sample(15000, random_state=42)

# 标签转数字
labels = df["category"].unique()
label2id = {label:i for i,label in enumerate(labels)}
id2label = {i:label for i,label in enumerate(labels)}
df["label"] = df["category"].map(label2id)

# 数据集划分
# 训练集 70% / 验证集 15% / 测试集 15%
train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)
test_dataset = Dataset.from_pandas(test_df)

print(f"训练集：{len(train_df)}")
print(f"验证集：{len(val_df)}")
print(f"测试集：{len(test_df)}")

# 加载BERT
model_name = "bert-base-chinese"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=len(labels),
    id2label=id2label,
    label2id=label2id
)

# 分词
def tokenize_func(examples):
    return tokenizer(
        examples["clean_title"],
        truncation=True,
        max_length=64
    )

tokenized_train = train_dataset.map(tokenize_func, batched=True)
tokenized_val = val_dataset.map(tokenize_func, batched=True)
tokenized_test = test_dataset.map(tokenize_func, batched=True)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# 评估指标
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc}


training_args = TrainingArguments(
    output_dir="./bert_news_classifier",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=10,
    learning_rate=2e-5,
    load_best_model_at_end=True
)

# 训练
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

trainer.train()

# 测试集评估
print("\n===== 测试集评估 =====")
test_results = trainer.evaluate(tokenized_test)
print(test_results)

# 分类报告
preds = trainer.predict(tokenized_test)
logits = preds.predictions
labels = preds.label_ids
pred_classes = np.argmax(logits, axis=-1)

print("\n===== 分类报告 =====")
print(classification_report(labels, pred_classes, target_names=list(id2label.values())))