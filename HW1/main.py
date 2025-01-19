import pandas as pd
import numpy as np
import nltk
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')
import re
from bs4 import BeautifulSoup

from sklearn.model_selection import train_test_split
import contractions


file_path = 'amazon_reviews_us_Office_Products_v1_00.tsv'

# 使用 Pandas 读取数据文件
data = pd.read_csv(file_path, sep='\t', usecols=['review_body', 'star_rating'])

data.rename(columns={'review_body': 'Review', 'star_rating': 'Rating'}, inplace=True)

RANDOM_NUM = 6
# 转换 Rating 列为数值类型
data['Rating'] = pd.to_numeric(data['Rating'], errors='coerce')

# 移除无效评分（NaN 值）
data = data.dropna(subset=['Rating'])
data = data.dropna(subset=['Review'])

# 映射评分为二元情感标签
# Rating > 3 映射为 1（正面），Rating <= 2 映射为 0（负面），Rating == 3 丢弃
data['Sentiment'] = data['Rating'].apply(lambda x: 1 if x > 3 else (0 if x <= 2 else None))

# 丢弃评分为 3 的评论
data = data.dropna(subset=['Sentiment'])

# 打印示例评论及对应评分
sample_reviews = data[['Review', 'Rating']].sample(3)
print("Three sample reviews: ")
print(sample_reviews)

# 统计评分的分布
rating_counts = data['Rating'].value_counts().sort_index()
print("\nStatistics of the ratings:")
print(rating_counts)

# 提取正面和负面评论各 100,000 条
positive_reviews = data[data['Sentiment'] == 1].sample(100000, random_state=RANDOM_NUM)
negative_reviews = data[data['Sentiment'] == 0].sample(100000, random_state=RANDOM_NUM)

random_reviews = positive_reviews.sample(n=5, random_state=RANDOM_NUM)
print(random_reviews[['Review', 'Rating']])

# # 合并正面和负面评论，形成新的数据集
# balanced_data = pd.concat([positive_reviews, negative_reviews]).reset_index(drop=True)

# # 按 80% 和 20% 的比例分割训练集和测试集
# train_data, test_data = train_test_split(
#     balanced_data, 
#     test_size=0.2, 
#     random_state=42,  # 确保分割的随机性可复现
#     stratify=balanced_data['Sentiment']  # 保证正负样本比例一致
# )


# 1. 转换为小写
positive_reviews['Review'] = positive_reviews['Review'].str.lower()
negative_reviews['Review'] = negative_reviews['Review'].str.lower()
# 2. 移除 HTML 标签
positive_reviews['Review'] = positive_reviews['Review'].apply(lambda x: BeautifulSoup(x, "html.parser").get_text())
negative_reviews['Review'] = negative_reviews['Review'].apply(lambda x: BeautifulSoup(x, "html.parser").get_text())

# 3. 移除 URL
positive_reviews['Review'] = positive_reviews['Review'].apply(lambda x: re.sub(r"http\S+|www\S+", "", x))
negative_reviews['Review'] = negative_reviews['Review'].apply(lambda x: re.sub(r"http\S+|www\S+", "", x))

# 4. 移除非字母字符
positive_reviews['Review'] = positive_reviews['Review'].apply(lambda x: re.sub(r"[^a-zA-Z\s]", "", x))
negative_reviews['Review'] = negative_reviews['Review'].apply(lambda x: re.sub(r"[^a-zA-Z\s]", "", x))


# 5. 删除多余的空格
positive_reviews['Review'] = positive_reviews['Review'].apply(lambda x: re.sub(r"\s+", " ", x).strip())
negative_reviews['Review'] = negative_reviews['Review'].apply(lambda x: re.sub(r"\s+", " ", x).strip())


# 6. 展开缩略形式
positive_reviews['Review'] = positive_reviews['Review'].apply(contractions.fix)
negative_reviews['Review'] = negative_reviews['Review'].apply(contractions.fix)


random_reviews = positive_reviews.sample(n=5, random_state=RANDOM_NUM)
print(random_reviews[['Review', 'Rating']])

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# # 数据清洗函数
# def preprocess_reviews(data):
#     # 初始化停用词和词形还原器
#     stop_words = set(stopwords.words('english'))
#     lemmatizer = WordNetLemmatizer()
    
#     # 定义处理逻辑
#     def clean_and_preprocess(review):
#         # 分词
#         # tokens = word_tokenize(review)
#         tokens = review.split(" ")
#         # 移除停用词并进行词形还原
#         processed_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
#         # 将处理后的词汇重新拼接为字符串
#         return ' '.join(processed_tokens)

#     # 应用到所有评论
#     data['Processed_Review'] = data['Review'].apply(clean_and_preprocess)
#     return data
# positive_reviews = preprocess_reviews(positive_reviews)
# negative_reviews = preprocess_reviews(negative_reviews)

def remove_stop_words(review):
    # tokens = word_tokenize(review)
    tokens = review.split(" ")
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

positive_reviews['Review'] = positive_reviews['Review'].apply(remove_stop_words)

random_reviews = positive_reviews.sample(n=5, random_state=RANDOM_NUM)
print(random_reviews[['Review', 'Rating']])

from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from nltk.corpus import wordnet


def get_wordnet_pos(tag):
    """
    将 nltk 的 POS tag 转换为 wordnet 的 pos 参数
    """
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def perform_lemmatization(review):
    lemmatizer = WordNetLemmatizer()
    tokens = review.split(" ")  # 或者你自己的分词方式
    pos_tags = nltk.pos_tag(tokens)
    
    lemmatized_tokens = []
    for word, tag in pos_tags:
        wordnet_pos = get_wordnet_pos(tag)
        lemmatized_tokens.append(lemmatizer.lemmatize(word, pos=wordnet_pos))
    
    return ' '.join(lemmatized_tokens)

positive_reviews['Review'] = positive_reviews['Review'].apply(perform_lemmatization)
random_reviews = positive_reviews.sample(n=5, random_state=RANDOM_NUM)
print(random_reviews[['Review', 'Rating']])

from sklearn.feature_extraction.text import TfidfVectorizer
# 合并正面和负面评论
all_reviews = pd.concat([positive_reviews, negative_reviews], ignore_index=True)

# 提取评论文本和标签
reviews = all_reviews['Review']  # 评论内容
labels = all_reviews['Sentiment']    # 二进制标签

# 初始化 TF-IDF 向量化器
tfidf_vectorizer = TfidfVectorizer(max_features=500, stop_words='english')  # 保留最多 500 个特征

# 提取 TF-IDF 特征
tfidf_features = tfidf_vectorizer.fit_transform(reviews)

# 转换为 DataFrame
tfidf_df = pd.DataFrame(tfidf_features.toarray(), columns=tfidf_vectorizer.get_feature_names_out())

# 添加标签到特征集
tfidf_df['Sentiment'] = labels.values

# 6. 拆分训练集和测试集（80% 训练, 20% 测试）
X_train, X_test, y_train, y_test = train_test_split(tfidf_features, labels, test_size=0.2, random_state=RANDOM_NUM)
print(X_train)
print(y_train)


from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 初始化并训练感知器模型
perceptron = Perceptron()
perceptron.fit(X_train, y_train)

# 在训练集上进行预测
y_train_pred = perceptron.predict(X_train)

# 在测试集上进行预测
y_test_pred = perceptron.predict(X_test)

# 计算指标
train_accuracy = accuracy_score(y_train, y_train_pred)
train_precision = precision_score(y_train, y_train_pred)
train_recall = recall_score(y_train, y_train_pred)
train_f1 = f1_score(y_train, y_train_pred)

test_accuracy = accuracy_score(y_test, y_test_pred)
test_precision = precision_score(y_test, y_test_pred)
test_recall = recall_score(y_test, y_test_pred)
test_f1 = f1_score(y_test, y_test_pred)

# 输出结果
print(f"Training Accuracy: {train_accuracy:.4f}")
print(f"Training Precision: {train_precision:.4f}")
print(f"Training Recall: {train_recall:.4f}")
print(f"Training F1-Score: {train_f1:.4f}")

print(f"Testing Accuracy: {test_accuracy:.4f}")
print(f"Testing Precision: {test_precision:.4f}")
print(f"Testing Recall: {test_recall:.4f}")
print(f"Testing F1-Score: {test_f1:.4f}")

from sklearn.svm import SVC

# 初始化并训练 SVM 模型
svm_model = SVC(kernel='linear', random_state=RANDOM_NUM)
svm_model.fit(X_train, y_train)

# 在训练集上进行预测
y_train_pred = svm_model.predict(X_train)

# 在测试集上进行预测
y_test_pred = svm_model.predict(X_test)

# 计算训练集指标
train_accuracy = accuracy_score(y_train, y_train_pred)
train_precision = precision_score(y_train, y_train_pred)
train_recall = recall_score(y_train, y_train_pred)
train_f1 = f1_score(y_train, y_train_pred)

# # 计算测试集指标
test_accuracy = accuracy_score(y_test, y_test_pred)
test_precision = precision_score(y_test, y_test_pred)
test_recall = recall_score(y_test, y_test_pred)
test_f1 = f1_score(y_test, y_test_pred)

# # 打印结果
print(f"Training Accuracy: {train_accuracy:.4f}")
print(f"Training Precision: {train_precision:.4f}")
print(f"Training Recall: {train_recall:.4f}")
print(f"Training F1-Score: {train_f1:.4f}")

print(f"Testing Accuracy: {test_accuracy:.4f}")
print(f"Testing Precision: {test_precision:.4f}")
print(f"Testing Recall: {test_recall:.4f}")
print(f"Testing F1-Score: {test_f1:.4f}")