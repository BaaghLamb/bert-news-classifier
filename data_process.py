import pandas as pd
from tqdm import tqdm
import re

def load_and_process_data(raw_file_path):
    data = []

    with open(raw_file_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in tqdm(f, desc="正在读取数据"):
            line = line.strip()
            if not line:
                continue

            parts = line.split('_!_')
            while len(parts) < 5:
                parts.append('')

            data.append({
                "id": parts[0],
                "cate_code": parts[1],
                "category": parts[2],
                "title": parts[3],
                "keywords": parts[4]
            })

    df = pd.DataFrame(data)


    df = df.drop_duplicates(subset=["id"])  # 去重

    def clean_text(text):
        # 清理所有会导致报错的字符
        text = text.replace('"', '')
        text = text.replace("'", "")
        text = text.replace(',', '，')
        text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9，。！？；：（）【】]', '', text)
        return text.strip()

    df["clean_title"] = df["title"].apply(clean_text)
    df = df[df["clean_title"].str.len() > 2]


    df.to_csv(
        "toutiao_clean_data.csv",
        index=False,
        encoding="utf-8-sig",
        escapechar='\\',  
        quoting=1
    )

    print("数据已保存为 toutiao_clean_data.csv")
    print(f"有效数据条数：{len(df)}")
    print("\n前5行数据预览：")
    print(df[["category", "clean_title"]].head())

if __name__ == "__main__":
    load_and_process_data("D:/NLP/toutiao-text-classfication-dataset-master/toutiao_cat_data.txt")