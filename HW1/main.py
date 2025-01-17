import pandas as pd
from sklearn.model_selection import train_test_split

# 数据读取
def load_and_prepare_data(file_path):
    # 使用 Pandas 读取数据文件
    data = pd.read_csv(file_path, sep='\t', usecols=['review_body', 'star_rating'])
    
    # 仅保留需要的字段，重命名方便后续处理
    data.rename(columns={'review_body': 'Review', 'star_rating': 'Rating'}, inplace=True)
    
    # 映射评分为二元情感标签
    # Rating > 3 映射为 1（正面），Rating <= 2 映射为 0（负面），Rating == 3 丢弃
    data['Sentiment'] = data['Rating'].apply(lambda x: 1 if x > 3 else (0 if x <= 2 else None))
    
    # 丢弃评分为 3 的评论
    data = data.dropna(subset=['Sentiment'])
    
    # 打印示例评论及对应评分
    sample_reviews = data[['Review', 'Rating']].sample(3)
    print("示例评论及评分：")
    print(sample_reviews)
    
    # 统计评分的分布
    rating_counts = data['Rating'].value_counts().sort_index()
    print("\n评分统计：")
    print(rating_counts)
    
    # 提取正面和负面评论各 100,000 条
    positive_reviews = data[data['Sentiment'] == 1].sample(100000, random_state=42)
    negative_reviews = data[data['Sentiment'] == 0].sample(100000, random_state=42)
    
    # 合并正面和负面评论，形成新的数据集
    balanced_data = pd.concat([positive_reviews, negative_reviews]).reset_index(drop=True)
    
    return balanced_data

# 数据集划分
def split_data(data):
    # 划分训练集和测试集，比例为 80%/20%
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42, stratify=data['Sentiment'])
    print("\n数据集划分完成：")
    print(f"训练集大小：{len(train_data)}")
    print(f"测试集大小：{len(test_data)}")
    return train_data, test_data

# 主函数
if __name__ == "__main__":
    # 替换为数据文件路径
    file_path = 'amazon_reviews_us_Office_Products_v1_00.tsv'
    
    # 加载和准备数据
    data = load_and_prepare_data(file_path)
    
    # 划分训练集和测试集
    train_data, test_data = split_data(data)
